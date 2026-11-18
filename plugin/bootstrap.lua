-- Starts Pixagon's Python side from inside Aseprite: finds or installs uv,
-- then launches the supervisor (python -m launcher.daemon) detached. The
-- pure helpers take the OS name ("windows" | "macos" | "linux"), so the
-- tests can ask about every OS from any OS; only B.ensure has side effects.
local pluginDir = ...

local B = {}
B.PYTHON = "3.12"          -- asked of uv; it downloads one if none fits
B.control = nil            -- set by main.lua: the control.lua instance

local UV_SH = "https://astral.sh/uv/install.sh"
local UV_PS = "https://astral.sh/uv/install.ps1"
-- Steam's runtime injects its own libraries into every child process, which
-- breaks system tools like curl (found in the spike); children drop them
local CLEAN_ENV = "env -u LD_LIBRARY_PATH -u LD_PRELOAD"
local EXIT_MARK = "pixagon-exit"  -- the installer's log ends with this line

-- This OS by name, from app.os (Aseprite 1.3+)
function B.osName()
  local o = app.os or {}
  if o.windows then return "windows" end
  if o.macos then return "macos" end
  return "linux"
end

-- Paths joined with this OS's separator
function B.join(osName, ...)
  return table.concat({ ... }, osName == "windows" and "\\" or "/")
end

-- Pixagon's per-user data folder; mirrors server/config.py:data_dir
function B.dataDir(osName, getenv)
  getenv = getenv or os.getenv
  local override = getenv("PIXAGON_DATA")
  if override and override ~= "" then return override end
  local home = getenv("HOME") or getenv("USERPROFILE") or ""
  if osName == "windows" then
    return B.join(osName, getenv("APPDATA") or home, "Pixagon")
  elseif osName == "macos" then
    return B.join(osName, home, "Library", "Application Support", "Pixagon")
  end
  local xdg = getenv("XDG_CONFIG_HOME")
  local base = (xdg and xdg ~= "") and xdg or B.join(osName, home, ".config")
  return B.join(osName, base, "pixagon")
end

-- The folder holding launcher/ and server/: the extension folder in a
-- release, the repo when plugin/ is linked into Aseprite from a checkout
function B.codeRoot(dir, osName, isFile)
  local function has(root)
    return isFile(B.join(osName, root, "launcher", "daemon.py"))
  end
  if has(dir) then return dir end
  local parent = dir:match("^(.+)[/\\][^/\\]+[/\\]?$")
  if parent and has(parent) then return parent end
  return nil
end

-- uv inside the data folder (the installer puts it in .tools or .tools/bin)
function B.uvPath(data, osName, isFile)
  local name = osName == "windows" and "uv.exe" or "uv"
  for _, p in ipairs({ B.join(osName, data, ".tools", name),
                       B.join(osName, data, ".tools", "bin", name) }) do
    if isFile(p) then return p end
  end
  return nil
end

-- POSIX shell quoting: single quotes, with embedded ones closed and escaped
local function shq(s) return "'" .. s:gsub("'", "'\\''") .. "'" end
-- PowerShell single-quoted string: '' stands for one quote
local function psq(s) return "'" .. s:gsub("'", "''") .. "'" end
-- cmd.exe argument quoting (Windows paths cannot contain ")
local function dq(s) return '"' .. s .. '"' end

-- A detached command running Astral's uv installer into <data>/.tools; its
-- log ends with "pixagon-exit <code>" so a failure shows without waiting
function B.uvInstallCommand(data, osName)
  local tools = B.join(osName, data, ".tools")
  local log = B.join(osName, data, "uv-install.log")
  if osName == "windows" then
    return 'start "" /B powershell -NoProfile -NonInteractive ' ..
           '-WindowStyle Hidden -ExecutionPolicy ByPass -Command "' ..
           "$env:UV_INSTALL_DIR=" .. psq(tools) .. "; " ..
           "$env:UV_NO_MODIFY_PATH='1'; " ..
           "irm " .. UV_PS .. " | iex *> " .. psq(log) .. "; " ..
           "Add-Content " .. psq(log) .. " ('" .. EXIT_MARK ..
           " ' + $LASTEXITCODE)\""
  end
  -- paths travel as environment variables, never inside the sh -c script
  return "nohup " .. CLEAN_ENV .. " UV_INSTALL_DIR=" .. shq(tools) ..
         " UV_NO_MODIFY_PATH=1 sh -c '(curl -LsSf " .. UV_SH ..
         " || wget -qO- " .. UV_SH .. ") | sh; echo " .. EXIT_MARK ..
         " $?' > " .. shq(log) .. " 2>&1 &"
end

-- A detached command starting the supervisor from the code root. On Windows
-- uvw.exe (uv without a console window) runs it when present
function B.supervisorCommand(root, data, uv, osName, isFile)
  local args = " run --no-project --python " .. B.PYTHON ..
               " --with websockets python -m launcher.daemon --data "
  if osName == "windows" then
    local uvw = uv:gsub("uv%.exe$", "uvw.exe")
    local exe = (isFile and isFile(uvw)) and uvw or uv
    return 'start "" /D ' .. dq(root) .. " " .. dq(exe) .. args .. dq(data)
  end
  -- the supervisor logs to daemon.log itself; this one catches uv's own
  -- output (the first-run Python download, or why it failed)
  return "cd " .. shq(root) .. " && nohup " .. CLEAN_ENV .. " " .. shq(uv) ..
         args .. shq(data) .. " > " ..
         shq(B.join(osName, data, "uv-run.log")) .. " 2>&1 &"
end

-- Last non-empty line of a text, clipped for a dialog label
function B.lastLine(text, max)
  local last
  for line in ((text or "") .. "\n"):gmatch("(.-)\r?\n") do
    if line:match("%S") then last = line end
  end
  last = last or ""
  if max and #last > max then last = last:sub(1, max - 3) .. "..." end
  return last
end

local function readFile(path)
  local f = io.open(path, "r")
  if not f then return nil end
  local text = f:read("a"); f:close()
  return text
end

-- Call fn every `interval` seconds until it returns true; onTimeout after
-- `limit` seconds. Without Timer (tests, odd builds) it gives up at once
local function every(interval, limit, fn, onTimeout)
  if not Timer then onTimeout(); return end
  local t0 = os.time()
  local timer
  timer = Timer{ interval = interval, ontick = function()
    if fn(os.time() - t0) then timer:stop(); return end
    if os.time() - t0 >= limit then
      timer:stop()
      onTimeout()
    end
  end }
  timer:start()
end

local waiters = nil  -- callbacks of the start in flight; nil when idle

-- Make sure the supervisor runs: ping it, else install uv if needed and
-- start it, then wait for its first answer. onStatus(text) narrates;
-- onReady() or onFail(message) ends it. Concurrent calls share one start.
function B.ensure(onStatus, onReady, onFail)
  if waiters then
    waiters[#waiters + 1] = { onReady, onFail }
    return
  end
  waiters = { { onReady, onFail } }
  local function finish(ok, msg)
    local list = waiters
    waiters = nil
    for _, w in ipairs(list) do
      if ok then w[1]() else w[2](msg) end
    end
  end

  local C = B.control
  local osName = B.osName()
  local data = B.dataDir(osName)
  local isFile = app.fs.isFile

  -- after spawning: ping each second until it answers (first run downloads
  -- Python, so allow minutes)
  local function waitForSupervisor()
    onStatus("Starting Pixagon (the first start downloads Python)...")
    local busy = false
    every(1, 300, function()
      if waiters == nil then return true end
      if not busy then
        busy = true
        C.ping(function() busy = false; finish(true) end,
               function() busy = false end)
      end
      return false
    end, function()
      finish(false, "Pixagon did not start; see " ..
                    B.join(osName, data, "daemon.log"))
    end)
  end

  local function spawn()
    local root = B.codeRoot(pluginDir, osName, isFile)
    if not root then
      finish(false, "launcher/ and server/ are missing from " .. pluginDir)
      return
    end
    local uv = B.uvPath(data, osName, isFile)
    os.execute(B.supervisorCommand(root, data, uv, osName, isFile))
    waitForSupervisor()
  end

  local function installUv()
    onStatus("Installing uv...")
    local log = B.join(osName, data, "uv-install.log")
    os.remove(log)
    os.execute(B.uvInstallCommand(data, osName))
    every(2, 180, function()
      if B.uvPath(data, osName, isFile) then spawn(); return true end
      local text = readFile(log) or ""
      if text:find(EXIT_MARK, 1, true) then
        -- the marker is the last line; the reason is the one before it
        local reason = text:gsub(EXIT_MARK:gsub("%-", "%%-") .. "[^\n]*", "")
        finish(false, "uv install failed: " .. B.lastLine(reason, 60))
        return true
      end
      return false
    end, function() finish(false, "uv install timed out; see " .. log) end)
  end

  C.ping(function() finish(true) end, function()
    app.fs.makeAllDirectories(data)
    if B.uvPath(data, osName, isFile) then spawn() else installUv() end
  end)
end

return B
