import { OpenAIError } from '../core/error';
import type { BodyInit, RequestInit } from './builtin-types';

/** Content codings the SDK can apply to request bodies. */
export type RequestCompression = 'zstd';

/**
 * The API rejects zstd bodies that decompress to more than this multiple of
 * their compressed size, so such bodies are sent uncompressed instead.
 */
const MAX_DECOMPRESSION_RATIO = 100;

/** The zstd level used by the `zstd` command line and the API's documented example. */
const ZSTD_LEVEL = 3;

/** The parts of `node:zlib` this module uses. zstd is available in Node.js 22.15+ and 23.8+. */
interface ZstdModule {
  zstdCompress: (
    input: Uint8Array,
    options: { params: Record<number, number>; pledgedSrcSize: number },
    callback: (error: Error | null, result: Uint8Array) => void,
  ) => void;
  constants: { ZSTD_c_compressionLevel: number };
}

type BuiltinModuleLoader = (id: 'node:zlib') => Partial<ZstdModule> | undefined;

/** Uncompressed sizes of the bodies this module produced, for request logging. */
const decodedLengths = new WeakMap<Uint8Array, number>();

function loadZstd(): ZstdModule | undefined {
  // SAFETY: `process` is optional outside Node.js, and its module loader is checked before it is called.
  const runtime = globalThis as { process?: { getBuiltinModule?: BuiltinModuleLoader } };
  const loader = runtime.process?.getBuiltinModule;
  // oxlint-disable-next-line anti-slop/no-runtime-typeof -- Host capability detection: node:zlib is reachable only through process.getBuiltinModule.
  if (typeof loader !== 'function') {
    return undefined;
  }
  let zlib: Partial<ZstdModule> | undefined;
  try {
    zlib = loader('node:zlib');
  } catch {
    return undefined;
  }
  const compress = zlib?.zstdCompress;
  const level = zlib?.constants?.ZSTD_c_compressionLevel;
  // oxlint-disable-next-line anti-slop/no-runtime-typeof -- Host capability detection: node:zlib has zstd only from Node.js 22.15 and 23.8.
  if (typeof compress !== 'function' || level === undefined || !Number.isInteger(level)) {
    return undefined;
  }
  return { zstdCompress: compress, constants: { ZSTD_c_compressionLevel: level } };
}

/** Resolves zstd support for an enabled option, or `undefined` when compression is disabled. */
function resolveCompression(
  value: RequestCompression | null | undefined,
  source: string,
): ZstdModule | undefined {
  if (value === undefined || value === null) {
    return undefined;
  }
  if (value !== 'zstd') {
    throw new OpenAIError(`${source} must be 'zstd' or null, but received ${JSON.stringify(value)}.`);
  }
  const zlib = loadZstd();
  if (!zlib) {
    throw new OpenAIError(
      `${source} 'zstd' needs a runtime whose node:zlib supports zstd, such as Node.js 22.15.0 or later. Omit ${source} in this runtime.`,
    );
  }
  return zlib;
}

/**
 * Validates a `requestCompression` option, and checks that this runtime can apply it.
 *
 * @param value The configured value; `undefined` and `null` disable compression.
 * @param source The option name used in error messages.
 * @throws {OpenAIError} If the value is not `'zstd'`, `null` or `undefined`, or if the
 * runtime's `node:zlib` has no zstd support.
 */
export function validateRequestCompression(
  value: RequestCompression | null | undefined,
  source: string,
): void {
  resolveCompression(value, source);
}

function isResponsesCreate(method: string, path: string): boolean {
  const end = path.search(/[?#]/u);
  return method.toLowerCase() === 'post' && (end === -1 ? path : path.slice(0, end)) === '/responses';
}

function isJSONContentType(value: string | null): boolean {
  return value?.split(';')[0]?.trim().toLowerCase() === 'application/json';
}

function zstdCompress(zlib: ZstdModule, input: Uint8Array): Promise<Uint8Array> {
  // oxlint-disable-next-line promise/avoid-new -- node:zlib's zstd functions are callback-based.
  return new Promise((resolve, reject) => {
    zlib.zstdCompress(
      input,
      {
        params: { [zlib.constants.ZSTD_c_compressionLevel]: ZSTD_LEVEL },
        // Records the uncompressed size in the frame header, as the `zstd` command does.
        pledgedSrcSize: input.byteLength,
      },
      // oxlint-disable-next-line promise/prefer-await-to-callbacks -- This callback is the Promise bridge.
      (error, result) => (error ? reject(error) : resolve(result)),
    );
  });
}

/**
 * Compresses the JSON body of a `POST /responses` request when compression is enabled.
 *
 * The body is left unchanged when compression is disabled, the request is not a
 * Responses create call, the body is not a JSON string, the caller already set
 * `Content-Encoding`, the request is already aborted, compression would not make the
 * body smaller, or the compressed body would exceed the API's decompression-ratio limit. Otherwise
 * `Content-Encoding: zstd` is set on `headers`, and the compressed bytes are returned.
 *
 * @param request.compression The effective request or client option.
 * @param request.headers The final request headers; `Content-Encoding` is added to them.
 * @param request.signal The request's abort signal; an aborted request is not compressed.
 * @returns The body to send.
 * @throws {OpenAIError} If the option is invalid or this runtime has no zstd support.
 */
export async function compressRequestBody<Body>(request: {
  compression: RequestCompression | null | undefined;
  method: string;
  path: string;
  headers: Headers;
  body: Body;
  signal?: AbortSignal | null | undefined;
}): Promise<Body | BodyInit> {
  const { compression, method, path, headers, body, signal } = request;
  const zlib = resolveCompression(compression, 'requestCompression');
  if (
    !zlib ||
    // oxlint-disable-next-line anti-slop/no-runtime-typeof -- Only SDK-serialized JSON strings are compressed; other bodies pass through.
    typeof body !== 'string' ||
    !isResponsesCreate(method, path) ||
    !isJSONContentType(headers.get('content-type')) ||
    headers.has('content-encoding') ||
    signal?.aborted
  ) {
    return body;
  }

  const encoded = new TextEncoder().encode(body);
  const compressed = await zstdCompress(zlib, encoded);
  if (
    compressed.byteLength >= encoded.byteLength ||
    encoded.byteLength > compressed.byteLength * MAX_DECOMPRESSION_RATIO
  ) {
    return body;
  }

  // Copy into a plain Uint8Array that owns its buffer, rather than a Node.js Buffer view.
  const wire = new Uint8Array(compressed);
  decodedLengths.set(wire, encoded.byteLength);
  headers.set('content-encoding', 'zstd');
  return wire;
}

/**
 * Describes a body produced by {@link compressRequestBody} for debug logs, without its contents.
 *
 * @returns A summary, or `undefined` if the SDK did not compress this body.
 */
export function compressedRequestBodySummary(
  body: RequestInit['body'],
): { type: 'zstd'; length: number; decodedLength: number } | undefined {
  if (!(body instanceof Uint8Array)) {
    return undefined;
  }
  const decodedLength = decodedLengths.get(body);
  return decodedLength === undefined ? undefined : { type: 'zstd', length: body.byteLength, decodedLength };
}
