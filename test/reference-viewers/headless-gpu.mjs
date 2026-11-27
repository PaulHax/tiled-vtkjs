import { spawn } from "node:child_process";
import { once } from "node:events";
import { createServer } from "node:net";
import { release } from "node:os";
import { createInterface } from "node:readline";

const windowsShell = (script) =>
  spawn(
    "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe",
    [
      "-NoProfile",
      "-NonInteractive",
      "-EncodedCommand",
      Buffer.from(script, "utf16le").toString("base64"),
    ],
    { stdio: ["pipe", "pipe", "pipe"] },
  );

const runWindowsChrome = String.raw`
$ErrorActionPreference = 'Stop'
$chrome = Join-Path $env:ProgramFiles 'Google\Chrome\Application\chrome.exe'
if (-not (Test-Path -LiteralPath $chrome)) { throw "Google Chrome not found at $chrome" }
$temp = (Resolve-Path -LiteralPath $env:TEMP).Path
$profile = Join-Path $temp ('tiled-vtkjs-benchmark-' + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $profile | Out-Null
try {
  $args = @('--headless=new', '--enable-gpu', '--disable-software-rasterizer', '--force_high_performance_gpu',
    '--remote-debugging-port=0', '--no-first-run', '--no-default-browser-check',
    '--disable-background-networking', '--disable-background-timer-throttling',
    '--disable-renderer-backgrounding', '--no-startup-window', ('--user-data-dir="' + $profile + '"'))
  $chromeProcess = Start-Process -FilePath $chrome -ArgumentList $args -WindowStyle Hidden -PassThru
  $portFile = Join-Path $profile 'DevToolsActivePort'
  $deadline = [DateTime]::UtcNow.AddSeconds(15)
  while (-not (Test-Path -LiteralPath $portFile)) {
    if ($chromeProcess.HasExited -or [DateTime]::UtcNow -ge $deadline) { throw 'Headless Chrome did not start DevTools' }
    Start-Sleep -Milliseconds 50
  }
  $lines = Get-Content -LiteralPath $portFile
  [Console]::WriteLine(('CHROME {0} {1}' -f $lines[0], $lines[1]))
  [Console]::ReadLine() | Out-Null
} finally {
  if ($chromeProcess -and -not $chromeProcess.HasExited) {
    [void]$chromeProcess.WaitForExit(3000)
    if (-not $chromeProcess.HasExited) { Stop-Process -Id $chromeProcess.Id }
  }
  $cleanup = (Resolve-Path -LiteralPath $profile -ErrorAction SilentlyContinue).Path
  if ($cleanup -and (Split-Path -Parent $cleanup) -eq $temp) {
    Remove-Item -LiteralPath $cleanup -Recurse -Force -ErrorAction SilentlyContinue
  }
}
`;

const endpointFrom = async (process) => {
  let error = "";
  process.stderr.on("data", (chunk) => {
    error += chunk;
  });
  const output = createInterface({ input: process.stdout });
  try {
    for await (const line of output) {
      const match = /^CHROME (\d+) (\/devtools\/browser\/[\w-]+)$/.exec(line);
      if (match) return { port: Number(match[1]), path: match[2] };
    }
    throw new Error(
      error.trim() || "Headless Windows Chrome exited during startup",
    );
  } finally {
    output.close();
  }
};

const tunnel = (port) =>
  windowsShell(`
$ErrorActionPreference = 'Stop'
$client = [System.Net.Sockets.TcpClient]::new('127.0.0.1', ${port})
try {
  $stream = $client.GetStream()
  $send = [Console]::OpenStandardInput().CopyToAsync($stream)
  $receive = $stream.CopyToAsync([Console]::OpenStandardOutput())
  [void][System.Threading.Tasks.Task]::WaitAny(@($send, $receive))
} finally {
  $client.Dispose()
}
`);

async function launchWindowsChrome(chromium) {
  const chrome = windowsShell(runWindowsChrome);
  chrome.stdin.on("error", () => {});
  const closed = new Promise((resolve) => {
    chrome.once("close", resolve);
    chrome.once("error", resolve);
  });
  const sockets = new Set();
  const relays = new Set();
  let bridge;
  const stop = async () => {
    for (const socket of sockets) socket.destroy();
    for (const relay of relays) relay.stdin.end();
    if (bridge?.listening)
      await new Promise((resolve) => bridge.close(resolve));
    chrome.stdin.end();
    await closed;
  };
  try {
    const { port, path } = await Promise.race([
      endpointFrom(chrome),
      once(chrome, "error").then(([error]) => {
        throw error;
      }),
    ]);
    // WSL in NAT mode cannot reach Windows loopback directly. Keep DevTools
    // on loopback in both environments and forward the bytes through stdio.
    bridge = createServer((socket) => {
      sockets.add(socket);
      const relay = tunnel(port);
      relays.add(relay);
      const close = () => {
        socket.destroy();
        relay.stdin.end();
      };
      relay.once("error", close);
      relay.once("close", () => {
        relays.delete(relay);
        close();
      });
      relay.stderr.resume();
      relay.stdin.on("error", close);
      socket.once("error", close);
      socket.once("close", () => {
        sockets.delete(socket);
        close();
      });
      socket.pipe(relay.stdin);
      relay.stdout.pipe(socket);
    });
    bridge.listen(0, "127.0.0.1");
    await once(bridge, "listening");
    const browser = await chromium.connectOverCDP(
      "ws://127.0.0.1:" + bridge.address().port + path,
      { timeout: 20_000 },
    );
    return {
      browser,
      close: async () => {
        try {
          await browser.close();
        } finally {
          await stop();
        }
      },
    };
  } catch (error) {
    await stop();
    throw error;
  }
}

export async function launchHeadlessGpu(chromium) {
  if (process.platform === "linux" && /microsoft/i.test(release()))
    return launchWindowsChrome(chromium);
  const browser = await chromium.launch({
    channel: "chromium",
    headless: true,
    args: [
      "--enable-gpu",
      "--disable-software-rasterizer",
      "--disable-dev-shm-usage",
    ],
  });
  return { browser, close: () => browser.close() };
}
