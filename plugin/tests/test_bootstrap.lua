-- Plain-Lua tests for bootstrap.lua and setup.lua's pure helpers: run with
-- `lua5.4 plugin/tests/test_bootstrap.lua` from the repo root.
local failures = 0
local function eq(got, want, what)
  if got ~= want then
    failures = failures + 1
    print(string.format("FAIL %s\n  got:  %s\n  want: %s",
                        what, tostring(got), tostring(want)))
  end
end
local function has(text, part, what)
  if not tostring(text):find(part, 1, true) then
    failures = failures + 1
    print(string.format("FAIL %s\n  %q\n  lacks %q", what, tostring(text), part))
  end
end
local function lacks(text, part, what)
  if tostring(text):find(part, 1, true) then
    failures = failures + 1
    print(string.format("FAIL %s\n  %q\n  has %q", what, tostring(text), part))
  end
end

app = { os = {}, fs = {} }
local B = assert(loadfile("plugin/bootstrap.lua"))("/ext/pixagon")
local S = assert(loadfile("plugin/setup.lua"))(nil, B, {}, "/data")

-- getenv from a table
local function env(t) return function(k) return t[k] end end
-- isFile true only for the listed paths
local function files(list)
  local set = {}
  for _, p in ipairs(list) do set[p] = true end
  return function(p) return set[p] == true end
end

-- ---------------------------------------------------------------- data dir
eq(B.dataDir("linux", env{ HOME = "/home/ada" }), "/home/ada/.config/pixagon",
   "linux data dir")
eq(B.dataDir("linux", env{ HOME = "/home/ada", XDG_CONFIG_HOME = "/cfg" }),
   "/cfg/pixagon", "linux follows XDG")
eq(B.dataDir("linux", env{ HOME = "/home/ada", XDG_CONFIG_HOME = "" }),
   "/home/ada/.config/pixagon", "empty XDG ignored, as in Python")
eq(B.dataDir("macos", env{ HOME = "/Users/ada" }),
   "/Users/ada/Library/Application Support/Pixagon", "macos data dir")
eq(B.dataDir("windows", env{ APPDATA = [[C:\Users\A B\AppData\Roaming]] }),
   [[C:\Users\A B\AppData\Roaming\Pixagon]], "windows data dir")
eq(B.dataDir("linux", env{ HOME = "/h", PIXAGON_DATA = "/elsewhere" }),
   "/elsewhere", "PIXAGON_DATA override")

-- ---------------------------------------------------------------- code root
eq(B.codeRoot("/ext/pixagon", "linux",
              files{ "/ext/pixagon/launcher/daemon.py" }),
   "/ext/pixagon", "release: code inside the extension")
eq(B.codeRoot("/repo/plugin", "linux", files{ "/repo/launcher/daemon.py" }),
   "/repo", "checkout: code beside plugin/")
eq(B.codeRoot([[C:\x\plugin]], "windows", files{ [[C:\x\launcher\daemon.py]] }),
   [[C:\x]], "checkout on windows")
eq(B.codeRoot("/nowhere/plugin", "linux", files{}), nil, "no code anywhere")

-- ---------------------------------------------------------------- uv
eq(B.uvPath("/d", "linux", files{ "/d/.tools/uv" }), "/d/.tools/uv", "uv in .tools")
eq(B.uvPath("/d", "linux", files{ "/d/.tools/bin/uv" }), "/d/.tools/bin/uv",
   "uv in .tools/bin")
eq(B.uvPath([[C:\d]], "windows", files{ [[C:\d\.tools\uv.exe]] }),
   [[C:\d\.tools\uv.exe]], "uv.exe on windows")
eq(B.uvPath("/d", "linux", files{}), nil, "no uv")

local unix = B.uvInstallCommand("/home/a b/it's", "linux")
has(unix, "env -u LD_LIBRARY_PATH -u LD_PRELOAD", "installer drops Steam libs")
has(unix, [[UV_INSTALL_DIR='/home/a b/it'\''s/.tools']], "tools dir quoted")
has(unix, "UV_NO_MODIFY_PATH=1", "shell profile left alone")
has(unix, "|| wget", "wget fallback")
has(unix, "pixagon-exit", "exit marker")
eq(unix:sub(-1), "&", "installer detached")

local win = B.uvInstallCommand([[C:\Users\O'Neil\AppData\Roaming\Pixagon]],
                               "windows")
has(win, 'start "" /B powershell', "windows installer detached")
has(win, "-WindowStyle Hidden", "no installer window")
has(win, [[$env:UV_INSTALL_DIR='C:\Users\O''Neil\AppData\Roaming\Pixagon\.tools']],
    "powershell quote doubled")
lacks(win, "env -u", "no POSIX env on windows")

-- ---------------------------------------------------------------- supervisor
local sup = B.supervisorCommand("/ext/pixa gon", "/data dir", "/data dir/.tools/uv",
                                "macos")
has(sup, "cd '/ext/pixa gon' && nohup env -u LD_LIBRARY_PATH", "unix cd + clean env")
has(sup, "'/data dir/.tools/uv' run --no-project --python 3.12 --with websockets",
    "uv run with only websockets")
has(sup, "python -m launcher.daemon --data '/data dir'", "daemon gets the data dir")
eq(sup:sub(-1), "&", "supervisor detached")

local winSup = B.supervisorCommand([[C:\ext]], [[C:\D]], [[C:\D\.tools\uv.exe]],
                                   "windows", files{ [[C:\D\.tools\uvw.exe]] })
has(winSup, [[start "" /D "C:\ext" "C:\D\.tools\uvw.exe" run]], "uvw preferred")
has(winSup, [[--data "C:\D"]], "windows data arg")
local noUvw = B.supervisorCommand([[C:\ext]], [[C:\D]], [[C:\D\.tools\uv.exe]],
                                  "windows", files{})
has(noUvw, [["C:\D\.tools\uv.exe" run]], "falls back to uv.exe")

-- ---------------------------------------------------------------- lastLine
eq(B.lastLine("a\nb\n\n"), "b", "last non-empty line")
eq(B.lastLine("one\r\ntwo\r\n"), "two", "CRLF")
eq(B.lastLine(nil), "", "nil text")
eq(B.lastLine("abcdefghij", 6), "abc...", "clipped")

-- ---------------------------------------------------------------- setup view
eq(S.clip("short"), "short", "short line untouched")
eq(#S.clip(string.rep("x", 100)), 56, "long line clipped to width")
eq(S.clip("/a/very/long/path/here", 12, true), "...path/here", "path keeps tail")

-- first values: saved settings win, junk falls back to the defaults
local init = S.initial({ provider = "openrouter", openrouter_api_key = "sk-1" }, nil)
eq(init.provider, "openrouter", "saved provider")
eq(init.local_model, S.DEFAULTS.local_model, "default local model")
eq(init.keyState, "API key: saved", "saved key noted, never shown")
eq(S.initial(nil, "sk-env").keyState, "API key: from OPENROUTER_API_KEY",
   "env key wins")
eq(S.initial({ provider = "bogus" }).provider, "local", "bad provider")

-- the full catalog is listed; planned models are marked, not hidden
local catalog = {
  { id = "a/klein", name = "FLUX.2 Klein", params = "4B", status = "supported",
    vram = { bf16 = 11.5, offload = 8 } },
  { id = "b/schnell", name = "Schnell", params = "12B", status = "planned" },
}
local models = S.localModels(catalog, "a/klein")
eq(#models, 2, "every catalog model listed")
eq(S.names(models)[2], "Schnell (12B) (planned)", "planned marked")
eq(S.byName(models, "FLUX.2 Klein (4B)").id, "a/klein", "name back to id")
eq(S.byName(models, "Schnell (12B) (planned)").planned, true, "planned flag")
local custom = S.localModels(catalog, "me/my-finetune")
eq(custom[3].id, "me/my-finetune", "a custom repo stays selectable")
eq(S.modelInfo(models[1]), "~11.5 GB VRAM, 8 GB with offload (picked automatically)",
   "VRAM line")
eq(S.modelInfo(models[2]), "Not supported yet.", "planned info")
eq(S.modelInfo(custom[3]), "Custom Hugging Face repo.", "custom info")

-- install rows and the line under the button
local rows = {
  { id = "uv", label = "uv", state = "ok", detail = "0.13", required = true },
  { id = "venv", label = "Virtual environment", state = "missing",
    detail = "missing", required = true },
  { id = "torch", label = "PyTorch", state = "missing", required = false },
  { id = "model", label = "Model", state = "missing", required = true,
    step_state = "running", detail = "x" },
}
eq(#S.requiredRows(rows), 3, "optional rows hidden")
eq(S.itemLine(rows[1]), "[ok] uv: 0.13", "ok row")
eq(S.itemLine(rows[4]), "[...] Model: x", "running row")
eq(S.itemLine{ id = "deps", label = "Deps", state = "missing", detail = "missing",
               required = true, step_state = "failed", step_detail = "see the log" },
   "[FAIL] Deps: see the log", "failed row shows why")
eq(table.concat(S.installIds(rows), ","), "venv,model", "required installables")
eq(S.summary{ items = rows }, "Install adds: venv, model", "what Install will do")
eq(S.summary{ running = true, items = rows }:sub(1, 13), "Installing...", "running")
eq(S.summary{ items = { { id = "apikey", state = "blocked", required = true } } },
   "Add your OpenRouter API key above and save it.", "key first")
eq(S.summary{ items = { rows[1] } }, "All set. Start the server from the Pixagon panel.",
   "done points to the panel")
local tail = S.tail("a\n\nb\nc\nd\n", 3)
eq(table.concat(tail, ","), "b,c,d", "last three log lines")
eq(#S.tail(nil, 3), 0, "no log")

eq(S.settled(nil), false, "no view yet")
eq(S.settled{ setup = { checking = true, items = { {} } } }, false, "still checking")
eq(S.settled{ setup = { checking = false, items = {} } }, false, "no rows yet")
eq(S.settled{ setup = { checking = false, items = { {} } } }, true, "settled")

if failures > 0 then
  print(failures .. " failure(s)")
  os.exit(1)
end
print("test_bootstrap ok")
