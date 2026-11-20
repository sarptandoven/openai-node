import { once } from 'node:events';
import { createServer } from 'node:http';
import type { ServerResponse } from 'node:http';
import type { Socket } from 'node:net';
import OpenAI from 'openai';
import { expect, test, vi } from 'vitest';

const params = {
  model: 'gpt-synthetic',
  messages: [{ role: 'user' as const, content: 'hello' }],
  stream: true as const,
};

function chunk(content: string, finishReason: 'stop' | null = null): string {
  const event = {
    id: 'chatcmpl_synthetic',
    object: 'chat.completion.chunk',
    created: 0,
    model: params.model,
    choices: [{ index: 0, delta: { role: 'assistant', content }, finish_reason: finishReason }],
  };
  return `data: ${JSON.stringify(event)}\n\n`;
}

async function withCompletedSSEServer(
  afterDone: (response: ServerResponse) => void,
  run: (client: OpenAI, sockets: Socket[]) => Promise<void>,
): Promise<void> {
  const sockets: Socket[] = [];
  const server = createServer((request, response) => {
    request.resume();
    request.on('end', () => {
      response.writeHead(200, { 'content-type': 'text/event-stream' });
      response.write(`${chunk('Hello')}${chunk(' world')}${chunk('', 'stop')}data: [DONE]\n\n`);
      afterDone(response);
    });
  });
  server.on('connection', (socket) => sockets.push(socket));

  try {
    const listening = once(server, 'listening');
    server.listen(0, '127.0.0.1');
    await listening;
    const address = server.address();
    if (!address || typeof address === 'string') {
      throw new Error('Expected a loopback TCP server address');
    }
    const client = new OpenAI({
      apiKey: 'sk-synthetic-keep-alive',
      baseURL: `http://127.0.0.1:${address.port}/v1`,
      maxRetries: 0,
      logLevel: 'off',
    });
    await run(client, sockets);
  } finally {
    if (server.listening) {
      const closed = once(server, 'close');
      server.close();
      server.closeAllConnections();
      await closed;
    }
  }
}

test('reuses the connection when the response ends shortly after [DONE]', async () => {
  await withCompletedSSEServer(
    (response) => {
      setTimeout(() => response.end(), 5);
    },
    async (client, sockets) => {
      const streamOnce = async () => {
        const { data: stream, response } = await client.chat.completions.create(params).withResponse();
        let text = '';
        for await (const event of stream) {
          text += event.choices[0]?.delta.content ?? '';
        }

        expect(text).toBe('Hello world');
        expect(stream.controller.signal.aborted).toBe(false);
        // The connection returns to the pool when the response ends. The unread body is released
        // shortly after that, so waiting for its release orders the next request after that end.
        await vi.waitFor(() => expect(response.body?.locked).toBe(false), { timeout: 3000 });
      };

      await streamOnce();
      await streamOnce();
      await streamOnce();
      expect(sockets).toHaveLength(1);
    },
  );
});

test('closes a connection that the server keeps open after [DONE]', async () => {
  await withCompletedSSEServer(
    () => {},
    async (client, sockets) => {
      const stream = await client.chat.completions.create(params);
      let text = '';
      for await (const event of stream) {
        text += event.choices[0]?.delta.content ?? '';
      }

      expect(text).toBe('Hello world');
      expect(sockets).toHaveLength(1);
      await vi.waitFor(() => expect(sockets[0]?.destroyed).toBe(true), { timeout: 3000 });
      expect(stream.controller.signal.aborted).toBe(false);
    },
  );
});
