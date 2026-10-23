// Node-side helpers for model metadata (host info, git commit).
import os from 'node:os';
import { execSync } from 'node:child_process';
export const hostInfo = () => ({ os: os.platform(), arch: os.arch(), cpus: os.cpus().length, cpu: os.cpus()[0]?.model || '' });
export const gitCommit = () => { try { return execSync('git rev-parse --short HEAD', { stdio: ['ignore', 'pipe', 'ignore'] }).toString().trim(); } catch { return null; } };
