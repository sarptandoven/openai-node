import { once } from 'node:events';
import { createServer } from 'node:http';
import type { IncomingMessage, ServerResponse } from 'node:http';
import * as zlib from 'node:zlib';
import { vi } from 'vitest';

import OpenAI from 'openai';
import { APIUserAbortError, OpenAIError } from 'openai/core/error';
import { createProvider } from 'openai/internal/provider';
import type { FinalRequestOptions } from 'openai/internal/request-options';
import type { Response as APIResponse, ResponseStreamEvent } from 'openai/resources/responses/responses';

// zstd is in node:zlib from Node.js 22.15 and 23.8.
const describeWithZstd = typeof zlib.zstdDecompressSync === 'function' ? describe : describe.skip;

type Body = RequestInit['body'];

interface Captured {
  url: string;
  headers: Headers;
  body: Body;
}

function makeResponse(overrides: Partial<APIResponse> = {}): APIResponse {
  // SAFETY: This synthetic wire response uses nullable legacy metadata defaults; these tests only inspect requests.
  return {
    id: 'resp_123',
    object: 'response',
    created_at: 1,
    error: null,
    incomplete_details: null,
    instructions: null,
    metadata: null,
    model: 'gpt-test',
    output: [],
    output_text: '',
    parallel_tool_calls: false,
    status: 'completed',
    temperature: null,
    tool_choice: 'auto',
    tools: [],
    top_p: null,
    max_output_tokens: null,
    previous_response_id: null,
    reasoning: { effort: null, summary: null },
    service_tier: null,
    store: true,
    text: { format: { type: 'text' }, verbosity: null },
    truncation: 'disabled',
    usage: null,
    user: null,
    ...overrides,
  } as APIResponse;
}

function jsonReply(): Response {
  return Response.json(makeResponse());
}

function eventStream(): Response {
  const events: ResponseStreamEvent[] = [
    { type: 'response.created', sequence_number: 0, response: makeResponse({ status: 'in_progress' }) },
    { type: 'response.completed', sequence_number: 1, response: makeResponse() },
  ];
  return new Response(events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join(''), {
    headers: { 'content-type': 'text/event-stream' },
  });
}

function capture(respond: (index: number) => Response = jsonReply) {
  const requests: Captured[] = [];
  const fetch = vi.fn(async (url: string | URL | Request, init?: RequestInit) => {
    requests.push({ url: String(url), headers: new Headers(init?.headers), body: init?.body });
    return respond(requests.length - 1);
  });
  return { requests, fetch };
}

function nth(requests: Captured[], index: number): Captured {
  const request = requests[index];
  if (!request) {
    throw new Error(`Expected request ${index}, but only ${requests.length} were sent.`);
  }
  return request;
}

function bytes(body: Body): Uint8Array {
  if (!(body instanceof Uint8Array)) {
    throw new Error(`Expected a compressed Uint8Array body, but received ${typeof body}.`);
  }
  return body;
}

function text(body: Body): string {
  if (typeof body !== 'string') {
    throw new Error(`Expected an uncompressed string body, but received ${typeof body}.`);
  }
  return body;
}

function decode(body: Body): string {
  return zlib.zstdDecompressSync(bytes(body)).toString('utf-8');
}

/** Deterministic text that zstd shrinks about four times, well within the API's ratio limit. */
function compressibleText(length: number): string {
  const words = ['alpha', 'bravo', 'charlie', 'delta', 'echo', 'foxtrot', 'golf', 'hotel', 'india'];
  const parts: string[] = [];
  // Park-Miller: a prime modulus avoids the short low-bit cycles of a power-of-two LCG.
  let state = 1;
  let size = 0;
  while (size < length) {
    state = (state * 48_271) % 2_147_483_647;
    const part = `${words[state % words.length]}-${Math.floor(state / words.length) % 1000}`;
    parts.push(part);
    size += part.length + 1;
  }
  return parts.join(' ');
}

function testProvider() {
  return createProvider({
    configure: () => ({
      name: 'test-provider',
      baseURL: 'https://provider.example.test/v1',
      prepareRequest(request) {
        const headers = new Headers(request.headers);
        headers.set('authorization', 'Bearer provider-token');
        request.headers = headers;
      },
    }),
  });
}

interface Prepared {
  body: Body;
  encoding: string | null;
}

/** Records what the protected `prepareRequest` hook receives, after the SDK builds the request. */
class RecordingOpenAI extends OpenAI {
  readonly prepared: Prepared[] = [];

  protected override async prepareRequest(
    request: RequestInit,
    context: { url: string; options: FinalRequestOptions },
  ): Promise<void> {
    await super.prepareRequest(request, context);
    this.prepared.push({
      body: request.body,
      encoding: new Headers(request.headers).get('content-encoding'),
    });
  }
}

const params = { model: 'gpt-test', input: compressibleText(64 * 1024) };

async function uncompressedBody(body: { model: string; input: string } = params): Promise<string> {
  const { requests, fetch } = capture();
  await new OpenAI({ apiKey: 'test-key', fetch }).responses.create(body);
  return text(nth(requests, 0).body);
}

describeWithZstd('zstd request compression', () => {
  test('compresses the Responses create body and keeps its JSON content type', async () => {
    const { requests, fetch } = capture();
    const client = new OpenAI({ apiKey: 'test-key', fetch, requestCompression: 'zstd' });

    await client.responses.create(params);

    const request = nth(requests, 0);
    const expected = await uncompressedBody();
    const body = bytes(request.body);
    expect(request.url).toBe('https://api.openai.com/v1/responses');
    expect(request.headers.get('content-encoding')).toBe('zstd');
    expect(request.headers.get('content-type')).toBe('application/json');
    expect(decode(body)).toBe(expected);
    expect(Object.getPrototypeOf(body)).toBe(Uint8Array.prototype);
    expect(body.byteLength).toBe(body.buffer.byteLength);
    expect(body.byteLength).toBeLessThan(expected.length / 2);
  });

  test('leaves requests uncompressed unless compression is enabled', async () => {
    const { requests, fetch } = capture();
    const client = new OpenAI({ apiKey: 'test-key', fetch });
    const expected = await uncompressedBody();

    await client.responses.create(params);
    await client.withOptions({ requestCompression: null }).responses.create(params);

    expect(requests.map((request) => request.headers.has('content-encoding'))).toEqual([false, false]);
    expect(requests.map((request) => request.body)).toEqual([expected, expected]);
  });

  test('lets request options enable or disable compression for one request', async () => {
    const { requests, fetch } = capture();
    const enabled = new OpenAI({ apiKey: 'test-key', fetch, requestCompression: 'zstd' });
    const disabled = new OpenAI({ apiKey: 'test-key', fetch });

    await enabled.responses.create(params, { requestCompression: null });
    await enabled.responses.create(params, { requestCompression: undefined });
    await disabled.responses.create(params, { requestCompression: 'zstd' });

    expect(requests.map((request) => request.headers.get('content-encoding'))).toEqual([
      null,
      'zstd',
      'zstd',
    ]);
    expect(nth(requests, 0).body).toBe(await uncompressedBody());
  });

  test('compresses beta, streaming, stream-helper and parse requests to the same endpoint', async () => {
    const { requests, fetch } = capture((index) => (index < 2 ? jsonReply() : eventStream()));
    const client = new OpenAI({ apiKey: 'test-key', fetch, requestCompression: 'zstd' });

    await client.beta.responses.create(params);
    await client.responses.parse(params);
    const stream = await client.responses.create({ ...params, stream: true });
    for await (const event of stream) {
      expect(event.type).toMatch(/^response\./u);
    }
    await client.responses.stream(params).finalResponse();

    expect(requests.map((request) => request.url)).toEqual([
      'https://api.openai.com/v1/responses?beta=true',
      'https://api.openai.com/v1/responses',
      'https://api.openai.com/v1/responses',
      'https://api.openai.com/v1/responses',
    ]);
    expect(requests.map((request) => request.headers.get('content-encoding'))).toEqual([
      'zstd',
      'zstd',
      'zstd',
      'zstd',
    ]);
    expect(requests.map((request) => JSON.parse(decode(request.body)))).toEqual([
      expect.objectContaining(params),
      expect.objectContaining(params),
      expect.objectContaining({ ...params, stream: true }),
      expect.objectContaining({ ...params, stream: true }),
    ]);
  });

  test('leaves other endpoints uncompressed', async () => {
    const { requests, fetch } = capture(() => Response.json({}));
    const client = new OpenAI({ apiKey: 'test-key', fetch, requestCompression: 'zstd' });
    const { input } = params;

    await client.chat.completions.create({
      model: 'gpt-test',
      messages: [{ role: 'user', content: input }],
    });
    await client.embeddings.create({ model: 'text-embedding-test', input });
    await client.responses.compact({ model: 'gpt-test', input });
    await client.responses.inputTokens.count({ model: 'gpt-test', input });
    await client.post('/responses/resp_123/cancel', { body: { input } });
    await client.post('https://example.test/v1/responses', { body: { input } });
    await client.responses.retrieve('resp_123');

    expect(requests).toHaveLength(7);
    expect(requests.some((request) => request.headers.has('content-encoding'))).toBe(false);
    expect(requests.map((request) => typeof request.body)).toEqual([
      'string',
      'string',
      'string',
      'string',
      'string',
      'string',
      'undefined',
    ]);
  });

  test('sends a body uncompressed when zstd would not shrink it or the API would reject its ratio', async () => {
    const { requests, fetch } = capture();
    const client = new OpenAI({ apiKey: 'test-key', fetch, requestCompression: 'zstd' });
    const tiny = { model: 'm', input: 'x' };
    const repetitive = { model: 'gpt-test', input: 'a'.repeat(2 * 1024 * 1024) };

    await client.responses.create(tiny);
    await client.responses.create(repetitive);
    await client.responses.create(params);

    expect(requests.map((request) => request.headers.get('content-encoding'))).toEqual([null, null, 'zstd']);
    expect(nth(requests, 0).body).toBe(await uncompressedBody(tiny));
    expect(nth(requests, 1).body).toBe(await uncompressedBody(repetitive));
    // Without the ratio check, the repetitive body would have been sent at more than 100 times.
    const encoded = new TextEncoder().encode(text(nth(requests, 1).body));
    expect(encoded.byteLength / zlib.zstdCompressSync(encoded).byteLength).toBeGreaterThan(100);
    const compressed = bytes(nth(requests, 2).body);
    expect(decode(compressed).length / compressed.byteLength).toBeLessThanOrEqual(100);
  });

  test("keeps a caller's own Content-Encoding and non-JSON bodies unchanged", async () => {
    const { requests, fetch } = capture();
    const client = new OpenAI({ apiKey: 'test-key', fetch, requestCompression: 'zstd' });
    const { input } = params;
    const binary = new TextEncoder().encode(input);

    await client.responses.create(params, { headers: { 'Content-Encoding': 'identity' } });
    await client.post('/responses', { body: input, headers: { 'content-type': 'text/plain' } });
    await client.post('/responses', { body: binary, headers: { 'content-type': 'application/json' } });

    expect(requests.map((request) => request.headers.get('content-encoding'))).toEqual([
      'identity',
      null,
      null,
    ]);
    expect(nth(requests, 0).body).toBe(await uncompressedBody());
    expect(nth(requests, 1).body).toBe(input);
    expect(nth(requests, 2).body).toBe(binary);
  });

  test('compresses JSON string bodies and JSON media types with parameters', async () => {
    const { requests, fetch } = capture();
    const client = new OpenAI({ apiKey: 'test-key', fetch, requestCompression: 'zstd' });
    const json = JSON.stringify(params);

    await client.post('/responses', {
      body: json,
      headers: { 'content-type': 'application/json; charset=utf-8' },
    });

    const request = nth(requests, 0);
    expect(request.headers.get('content-encoding')).toBe('zstd');
    expect(request.headers.get('content-type')).toBe('application/json; charset=utf-8');
    expect(decode(request.body)).toBe(json);
  });

  test('sends the same compressed body on every retry', async () => {
    const { requests, fetch } = capture((index) =>
      index === 0
        ? Response.json(
            { error: { message: 'unavailable' } },
            { status: 503, headers: { 'retry-after-ms': '1' } },
          )
        : jsonReply(),
    );
    const client = new OpenAI({ apiKey: 'test-key', fetch, requestCompression: 'zstd', maxRetries: 1 });
    const expected = await uncompressedBody();

    await client.responses.create(params);

    expect(requests.map((request) => request.headers.get('x-stainless-retry-count'))).toEqual(['0', '1']);
    expect(requests.map((request) => request.headers.get('content-encoding'))).toEqual(['zstd', 'zstd']);
    expect(requests.map((request) => decode(request.body))).toEqual([expected, expected]);
    expect(nth(requests, 1).body).toEqual(nth(requests, 0).body);
  });

  test('compresses before request hooks and providers sign or inspect the request', async () => {
    const providerSaw: Prepared[] = [];
    const { requests, fetch } = capture();
    const hooked = new RecordingOpenAI({ apiKey: 'test-key', fetch, requestCompression: 'zstd' });
    const provided = new OpenAI({
      fetch,
      requestCompression: 'zstd',
      provider: createProvider({
        configure: () => ({
          name: 'test-provider',
          baseURL: 'https://provider.example.test/v1',
          prepareRequest(request) {
            const headers = new Headers(request.headers);
            providerSaw.push({ body: request.body, encoding: headers.get('content-encoding') });
            headers.set('authorization', 'Bearer provider-token');
            request.headers = headers;
          },
        }),
      }),
    });
    const expected = await uncompressedBody();

    await hooked.responses.create(params);
    await provided.responses.create(params);

    const seen = [...hooked.prepared, ...providerSaw];
    expect(seen.map(({ encoding }) => encoding)).toEqual(['zstd', 'zstd']);
    expect(seen.map(({ body }) => body)).toEqual(requests.map(({ body }) => body));
    expect(seen.map(({ body }) => decode(body))).toEqual([expected, expected]);
    expect(nth(requests, 1).url).toBe('https://provider.example.test/v1/responses');
  });

  test('withOptions keeps the setting, like timeout, unless it is overridden', async () => {
    const { requests, fetch } = capture();
    const client = new OpenAI({ apiKey: 'test-key', fetch, requestCompression: 'zstd' });

    await client.withOptions({ timeout: 1000 }).responses.create(params);
    await client.withOptions({ baseURL: 'https://gateway.example.test/v1' }).responses.create(params);
    await client.withOptions({ provider: testProvider(), fetch }).responses.create(params);
    await client.withOptions({ requestCompression: null }).responses.create(params);
    await client
      .withOptions({ provider: testProvider(), fetch, requestCompression: null })
      .responses.create(params);

    expect(requests.map((request) => request.headers.get('content-encoding'))).toEqual([
      'zstd',
      'zstd',
      'zstd',
      null,
      null,
    ]);
    expect(requests.map((request) => new URL(request.url).origin)).toEqual([
      'https://api.openai.com',
      'https://gateway.example.test',
      'https://provider.example.test',
      'https://api.openai.com',
      'https://provider.example.test',
    ]);
  });

  test('does not compress a request whose signal is already aborted', async () => {
    const { fetch } = capture();
    const client = new RecordingOpenAI({ apiKey: 'test-key', fetch, requestCompression: 'zstd' });

    await expect(
      client.responses.create(params, { signal: AbortSignal.abort(new Error('stop')) }),
    ).rejects.toBeInstanceOf(APIUserAbortError);

    expect(fetch).not.toHaveBeenCalled();
    expect(client.prepared).toEqual([{ body: await uncompressedBody(), encoding: null }]);
  });

  test('debug logs describe a compressed body without its bytes', async () => {
    const debug = vi.fn();
    const logger = { debug, info: vi.fn(), warn: vi.fn(), error: vi.fn() };
    const { requests, fetch } = capture();
    const client = new OpenAI({
      apiKey: 'test-key',
      fetch,
      requestCompression: 'zstd',
      logLevel: 'debug',
      logger,
    });
    const expected = await uncompressedBody();

    await client.responses.create(params);

    const sending = debug.mock.calls.find(([message]) => String(message).includes('sending request'));
    expect(sending?.[1]).toMatchObject({
      options: {
        body: {
          type: 'zstd',
          length: bytes(nth(requests, 0).body).byteLength,
          decodedLength: expected.length,
        },
      },
    });
    expect(JSON.stringify(sending)).not.toContain('alpha');
  });

  test('sends a body that a real HTTP server decodes', async () => {
    const received: { encoding?: string; length?: string; type?: string; body: string }[] = [];
    const server = createServer((request: IncomingMessage, response: ServerResponse) => {
      const chunks: Buffer[] = [];
      request.on('data', (chunk: Buffer) => chunks.push(chunk));
      request.on('end', () => {
        let body: string;
        try {
          body = zlib.zstdDecompressSync(Buffer.concat(chunks)).toString('utf-8');
        } catch {
          // Like the API, answer invalid zstd data with a 400 rather than leaving the request open.
          response.statusCode = 400;
          response.end();
          return;
        }
        received.push({
          ...(request.headers['content-encoding'] && { encoding: request.headers['content-encoding'] }),
          ...(request.headers['content-length'] && { length: request.headers['content-length'] }),
          ...(request.headers['content-type'] && { type: request.headers['content-type'] }),
          body,
        });
        response.setHeader('content-type', 'application/json');
        response.end(JSON.stringify(makeResponse()));
      });
    });
    server.listen(0, '127.0.0.1');
    await once(server, 'listening');
    try {
      const address = server.address();
      if (address === null || typeof address === 'string') {
        throw new Error('Expected a TCP server address.');
      }
      const client = new OpenAI({
        apiKey: 'test-key',
        baseURL: `http://127.0.0.1:${address.port}/v1`,
        requestCompression: 'zstd',
        maxRetries: 0,
      });

      await expect(client.responses.create(params)).resolves.toMatchObject({ id: 'resp_123' });
    } finally {
      server.closeAllConnections();
      server.close();
      await once(server, 'close');
    }

    const expected = await uncompressedBody();
    expect(received).toEqual([
      {
        encoding: 'zstd',
        length: expect.stringMatching(/^[1-9]\d*$/u),
        type: 'application/json',
        body: expected,
      },
    ]);
  });

  test('compresses a large synthetic body', async () => {
    const { requests, fetch } = capture();
    const client = new OpenAI({ apiKey: 'test-key', fetch, requestCompression: 'zstd' });
    const large = { model: 'gpt-test', input: compressibleText(16 * 1024 * 1024) };

    await client.responses.create(large);

    const request = nth(requests, 0);
    expect(request.headers.get('content-encoding')).toBe('zstd');
    expect(decode(request.body)).toBe(JSON.stringify(large));
  });
});

function missingZstdMessage(source: string): string {
  return `${source} 'zstd' needs a runtime whose node:zlib supports zstd, such as Node.js 22.15.0 or later. Omit ${source} in this runtime.`;
}

const missingZstdLoaders: [string, (() => object) | undefined][] = [
  ['getBuiltinModule is missing', undefined],
  ['node:zlib has no zstd', () => ({ constants: {} })],
  [
    'node:zlib cannot be loaded',
    () => {
      throw new Error('no node:zlib');
    },
  ],
];

describe('requestCompression validation', () => {
  test('rejects unsupported values for the client and for a request', async () => {
    for (const value of ['gzip', 'ZSTD', true, '']) {
      // @ts-expect-error Only 'zstd' and null are accepted.
      expect(() => new OpenAI({ apiKey: 'test-key', requestCompression: value })).toThrow(
        new OpenAIError(
          `ClientOptions.requestCompression must be 'zstd' or null, but received ${JSON.stringify(value)}.`,
        ),
      );
    }

    const { fetch } = capture();
    const client = new OpenAI({ apiKey: 'test-key', fetch });
    await expect(
      // @ts-expect-error Only 'zstd' and null are accepted.
      client.responses.create(params, { requestCompression: 'br' }),
    ).rejects.toThrow(new OpenAIError(`requestCompression must be 'zstd' or null, but received "br".`));
    expect(fetch).not.toHaveBeenCalled();
  });

  test.each(missingZstdLoaders)('explains the runtime requirement when %s', async (_case, loader) => {
    const descriptor = Object.getOwnPropertyDescriptor(process, 'getBuiltinModule');
    const { fetch } = capture();
    const uncompressed = new OpenAI({ apiKey: 'test-key', fetch });
    Object.defineProperty(process, 'getBuiltinModule', { value: loader, configurable: true, writable: true });
    try {
      expect(() => new OpenAI({ apiKey: 'test-key', requestCompression: 'zstd' })).toThrow(
        new OpenAIError(missingZstdMessage('ClientOptions.requestCompression')),
      );
      expect(() => new OpenAI({ apiKey: 'test-key', requestCompression: null })).not.toThrow();
      await expect(uncompressed.responses.create(params, { requestCompression: 'zstd' })).rejects.toThrow(
        new OpenAIError(missingZstdMessage('requestCompression')),
      );
      await expect(uncompressed.responses.create(params)).resolves.toMatchObject({ id: 'resp_123' });
      expect(fetch).toHaveBeenCalledTimes(1);
    } finally {
      if (descriptor) {
        Object.defineProperty(process, 'getBuiltinModule', descriptor);
      } else {
        Reflect.deleteProperty(process, 'getBuiltinModule');
      }
    }
  });
});
