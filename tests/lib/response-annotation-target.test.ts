import OpenAI from 'openai';
import { accumulateResponse } from 'openai/lib/responses/ResponseAccumulator';
import { ResponseStream } from 'openai/lib/responses/ResponseStream';
import { ResponsesWebSocketAccumulator } from 'openai/lib/responses/responses-websocket-accumulator';
import type {
  Response,
  ResponseFunctionToolCall,
  ResponseFunctionWebSearch,
  ResponseOutputMessage,
  ResponseOutputText,
  ResponseOutputTextAnnotationAddedEvent,
  ResponseStreamEvent,
} from 'openai/resources/responses/responses';

type WireEvent = ResponseStreamEvent extends infer Event
  ? Event extends ResponseStreamEvent
    ? Omit<Event, 'sequence_number'>
    : never
  : never;

const encoder = new TextEncoder();

const webSearch: ResponseFunctionWebSearch = {
  type: 'web_search_call',
  id: 'ws_synthetic',
  status: 'completed',
  action: { type: 'search', query: 'synthetic query' },
};
const messageCitation: ResponseOutputText.URLCitation = {
  type: 'url_citation',
  url: 'https://example.test/message',
  title: 'Message source',
  start_index: 0,
  end_index: 4,
};
const functionCitation: ResponseOutputText.URLCitation = {
  type: 'url_citation',
  url: 'https://example.test/function',
  title: 'Function source',
  start_index: 0,
  end_index: 0,
};
const message: ResponseOutputMessage = {
  type: 'message',
  id: 'msg_synthetic',
  role: 'assistant',
  status: 'completed',
  content: [{ type: 'output_text', text: 'See source.', annotations: [messageCitation], logprobs: [] }],
};
const functionCall: ResponseFunctionToolCall = {
  type: 'function_call',
  id: 'fc_synthetic',
  call_id: 'call_synthetic',
  name: 'lookup',
  arguments: '{"query":"synthetic"}',
  status: 'completed',
};

function response(output: Response['output'], status: 'in_progress' | 'completed'): Response {
  return {
    id: 'resp_synthetic',
    object: 'response',
    access_programs: null,
    created_at: 1,
    error: null,
    incomplete_details: null,
    instructions: null,
    metadata: null,
    model: 'gpt-synthetic',
    output,
    output_text: '',
    parallel_tool_calls: true,
    status,
    temperature: null,
    tool_choice: 'auto',
    tools: [],
    top_p: null,
  };
}

// A hosted web search followed by a function call, where the API also emits a citation whose
// output_index and item_id identify the function call rather than a message.
const functionCitationEvent: Omit<ResponseOutputTextAnnotationAddedEvent, 'sequence_number'> = {
  type: 'response.output_text.annotation.added',
  output_index: 2,
  item_id: 'fc_synthetic',
  content_index: 0,
  annotation_index: 0,
  annotation: functionCitation,
};

const wireEvents: WireEvent[] = [
  { type: 'response.created', response: response([], 'in_progress') },
  { type: 'response.output_item.added', output_index: 0, item: { ...webSearch, status: 'in_progress' } },
  { type: 'response.web_search_call.searching', output_index: 0, item_id: webSearch.id },
  { type: 'response.web_search_call.completed', output_index: 0, item_id: webSearch.id },
  { type: 'response.output_item.done', output_index: 0, item: webSearch },
  {
    type: 'response.output_item.added',
    output_index: 1,
    item: { ...message, status: 'in_progress', content: [] },
  },
  {
    type: 'response.content_part.added',
    output_index: 1,
    item_id: message.id,
    content_index: 0,
    part: { type: 'output_text', text: '', annotations: [] },
  },
  {
    type: 'response.output_text.delta',
    output_index: 1,
    item_id: message.id,
    content_index: 0,
    delta: 'See source.',
    logprobs: [],
  },
  {
    type: 'response.output_text.annotation.added',
    output_index: 1,
    item_id: message.id,
    content_index: 0,
    annotation_index: 0,
    annotation: messageCitation,
  },
  { type: 'response.output_item.done', output_index: 1, item: message },
  {
    type: 'response.output_item.added',
    output_index: 2,
    item: { ...functionCall, arguments: '', status: 'in_progress' },
  },
  {
    type: 'response.function_call_arguments.delta',
    output_index: 2,
    item_id: 'fc_synthetic',
    delta: functionCall.arguments,
  },
  functionCitationEvent,
  {
    type: 'response.function_call_arguments.done',
    output_index: 2,
    item_id: 'fc_synthetic',
    arguments: functionCall.arguments,
  },
  { type: 'response.output_item.done', output_index: 2, item: functionCall },
  { type: 'response.completed', response: response([webSearch, message, functionCall], 'completed') },
];
const events = wireEvents.map(
  // SAFETY: Adding the wire-order sequence number restores the field omitted from each event variant.
  (event, sequence_number) => ({ ...event, sequence_number }) as ResponseStreamEvent,
);

function sseBody(): ReadableStream<Uint8Array> {
  const wire = events.map((event) => `event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`).join('');
  return new ReadableStream({
    start(controller) {
      controller.enqueue(encoder.encode(wire));
      controller.close();
    },
  });
}

function client(): OpenAI {
  return new OpenAI({
    apiKey: 'sk-synthetic-annotation-target',
    maxRetries: 0,
    fetch: async () => new Response(sseBody(), { headers: { 'content-type': 'text/event-stream' } }),
  });
}

function expectCompletedOutput(final: Response) {
  expect(final.output).toHaveLength(3);
  expect(final.output[1]).toMatchObject({
    type: 'message',
    content: [{ type: 'output_text', text: 'See source.', annotations: [messageCitation] }],
  });
  expect(final.output[2]).toMatchObject({ type: 'function_call', arguments: functionCall.arguments });
}

async function collectAnnotations(stream: ResponseStream<unknown>) {
  const annotations: unknown[] = [];
  stream.on('response.output_text.annotation.added', (event) => annotations.push(event.annotation));
  const final = await stream.finalResponse();
  return { final, annotations };
}

describe('response.output_text.annotation.added targeting a non-message output item', () => {
  test('lets responses.stream() finish and still emits both citations', async () => {
    const { final, annotations } = await collectAnnotations(
      client().responses.stream({ model: 'gpt-synthetic', input: 'hello' }),
    );

    expectCompletedOutput(final);
    expect(annotations).toEqual([messageCitation, functionCitation]);
  });

  test('lets ResponseStream.fromReadableStream() finish', async () => {
    const raw = await client().responses.create({ model: 'gpt-synthetic', input: 'hello', stream: true });
    const { final, annotations } = await collectAnnotations(
      ResponseStream.fromReadableStream(raw.toReadableStream()),
    );

    expectCompletedOutput(final);
    expect(annotations).toEqual([messageCitation, functionCitation]);
  });

  test('leaves an accumulateResponse() snapshot unchanged', () => {
    let snapshot: Response | undefined;
    let ignored = 0;
    for (const event of events) {
      if (event.type === functionCitationEvent.type && event.item_id === functionCitationEvent.item_id) {
        const before = structuredClone(snapshot);
        expect(accumulateResponse(event, snapshot)).toBe(snapshot);
        expect(snapshot).toEqual(before);
        ignored += 1;
      } else {
        snapshot = accumulateResponse(event, snapshot);
      }
    }
    expect(ignored).toBe(1);
    if (!snapshot) {
      throw new Error('Expected an accumulated response');
    }
    expectCompletedOutput(snapshot);
  });

  test('leaves a Responses WebSocket snapshot unchanged', () => {
    const accumulator = new ResponsesWebSocketAccumulator();
    for (const event of events) {
      if (event.type === 'response.completed') {
        const state = accumulator.current;
        if (state?.phase !== 'provisional') {
          throw new Error(`Expected provisional output, got ${state?.phase}`);
        }
        expect(state.snapshot.output[2]).toMatchObject({ type: 'function_call' });
        expect(state.snapshot.output[2]).not.toHaveProperty('content');
      }
      accumulator.add(event);
    }
    expect(accumulator.current).toMatchObject({ phase: 'terminal' });
  });

  test.each(['function_call', 'web_search_call'] as const)(
    'still rejects a mismatched item_id for a %s target',
    (type) => {
      const target = type === 'function_call' ? functionCall : webSearch;
      const snapshot = accumulateResponse({
        type: 'response.created',
        sequence_number: 0,
        response: response([target], 'in_progress'),
      });

      expect(() =>
        accumulateResponse(
          { ...functionCitationEvent, output_index: 0, item_id: 'foreign_item', sequence_number: 1 },
          snapshot,
        ),
      ).toThrow(`expected item_id '${target.id}', got 'foreign_item'`);
    },
  );

  test('keeps rejecting output text that targets a function call', () => {
    const snapshot = accumulateResponse({
      type: 'response.created',
      sequence_number: 0,
      response: response([functionCall], 'in_progress'),
    });
    const before = structuredClone(snapshot);

    expect(() =>
      accumulateResponse(
        {
          type: 'response.output_text.delta',
          sequence_number: 1,
          output_index: 0,
          item_id: 'fc_synthetic',
          content_index: 0,
          delta: 'injected',
          logprobs: [],
        },
        snapshot,
      ),
    ).toThrow("expected output item type 'message', got 'function_call'");
    expect(snapshot).toEqual(before);
  });
});
