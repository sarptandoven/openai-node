import { spawnSync } from 'node:child_process';
import { mkdirSync, mkdtempSync, rmSync, symlinkSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { spawnBashWithCommands } from './utils/bash';

function writeExecutable(file: string, contents: string): void {
  writeFileSync(file, contents, { mode: 0o755 });
}

// The fake launcher below is a POSIX shell script; Windows CI exercises the real Git Bash launcher.
const describePosix = process.platform === 'win32' ? describe.skip : describe;

describePosix('spawnBashWithCommands', () => {
  let fixture: string;
  let env: NodeJS.ProcessEnv;
  let commands: string;

  beforeEach(() => {
    fixture = mkdtempSync(path.join(tmpdir(), 'openai-node-bash-helper-'));
    const bash = spawnSync('bash', ['-c', 'printf %s "$BASH"'], { encoding: 'utf-8' }).stdout;
    const launcher = path.join(fixture, 'launcher');
    const system = path.join(fixture, 'system');
    // A directory name that Git Bash would mangle if it were passed as an argument.
    commands = path.join(fixture, "user's commands");
    for (const directory of [launcher, system, commands]) {
      mkdirSync(directory);
    }
    // Like Git for Windows' bin/bash.exe, this launcher puts its own tools ahead of the PATH it
    // inherits, and its tool directory holds the real bash as well.
    writeExecutable(
      path.join(launcher, 'bash'),
      `#!${bash}\nexport PATH=${JSON.stringify(system)}":$PATH"\nexec ${JSON.stringify(bash)} "$@"\n`,
    );
    symlinkSync(bash, path.join(system, 'bash'));
    writeExecutable(path.join(system, 'curl'), '#!/bin/sh\necho launcher curl\n');
    writeExecutable(path.join(commands, 'curl'), '#!/bin/sh\necho fixture curl\n');
    env = { ...process.env, PATH: `${launcher}${path.delimiter}${process.env['PATH']}` };
  });

  afterEach(() => {
    rmSync(fixture, { recursive: true, force: true });
  });

  test('keeps fixture commands ahead of a launcher that prepends its own tools', () => {
    const plain = spawnSync('bash', ['-c', 'curl'], { encoding: 'utf-8', env });
    expect(plain.stdout).toBe('launcher curl\n');

    const result = spawnBashWithCommands([commands], ['-c', 'curl; printf %s "$OPENAI_TEST_COMMAND_PATH"'], {
      encoding: 'utf-8',
      env,
    });

    expect(result.error).toBeUndefined();
    expect(result.status).toBe(0);
    expect(result.stdout).toBe('fixture curl\n');
  });

  test('requires a command directory', () => {
    expect(() => spawnBashWithCommands([], ['-c', 'true'], { encoding: 'utf-8' })).toThrow(
      'Expected at least one command directory',
    );
  });
});
