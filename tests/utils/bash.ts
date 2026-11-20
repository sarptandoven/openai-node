import { spawnSync } from 'node:child_process';
import type { SpawnSyncOptionsWithStringEncoding, SpawnSyncReturns } from 'node:child_process';

/**
 * Converts a Windows drive or UNC path to the form Git Bash uses (`C:\x` to `/c/x`, and backslashes to
 * forward slashes); other platforms keep the path unchanged.
 */
export function toBashPath(file: string): string {
  if (process.platform !== 'win32') {
    return file;
  }
  return file
    .replace(/^[A-Za-z]:[\\/]/u, (prefix) => `/${prefix.charAt(0).toLowerCase()}/`)
    .split('\\')
    .join('/');
}

/**
 * Runs `bash` with `commandDirectories` ahead of every other directory on PATH.
 *
 * Git for Windows' `bin/bash.exe` launcher puts its own tool directories ahead of the PATH it
 * inherits, so its `curl` and `sleep` would shadow fixture stubs supplied through `options.env`. The
 * directories are therefore prepended after Bash starts. `$BASH` is then the real
 * `usr/bin/bash.exe`, not the launcher, so running `args` with it keeps that order. The directories
 * travel in an environment variable, which is unset before `args` run, so `args` reach that Bash
 * unchanged.
 */
export function spawnBashWithCommands(
  commandDirectories: readonly string[],
  args: readonly string[],
  options: SpawnSyncOptionsWithStringEncoding,
): SpawnSyncReturns<string> {
  if (commandDirectories.length === 0) {
    throw new Error('Expected at least one command directory');
  }
  return spawnSync(
    'bash',
    [
      '-c',
      'PATH="$OPENAI_TEST_COMMAND_PATH:$PATH"; unset OPENAI_TEST_COMMAND_PATH; exec "$BASH" "$@"',
      'bash',
      ...args,
    ],
    {
      ...options,
      env: {
        ...(options.env ?? process.env),
        OPENAI_TEST_COMMAND_PATH: commandDirectories.map(toBashPath).join(':'),
      },
    },
  );
}
