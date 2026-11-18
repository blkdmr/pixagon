-- The Setup dialog: choose the backend and install what it needs. Starting
-- the server lives on the panel (S.startServer), so Setup is only about
-- getting ready. The provider and local model save as soon as they are
-- picked; the OpenRouter fields save with their own button (or Install).
-- The fields fill once from config.json and the model catalog on disk;
-- polling the supervisor (once a second) only repaints the Install part.
local pluginDir, bootstrap, control, dataDir = ...

-- Aseprite's json.decode returns userdata (indexable, with # and ipairs),
-- not a table; plain-Lua tests return tables. Either is a decoded object.
local function isObject(v)
  return type(v) == "table" or type(v) == "userdata"
end

local S = {}
local CLIP = 56           -- characters per dynamic line
local MAX_ROWS = 5        -- required rows: uv, venv, torch, deps, model/apikey
local LOG_LINES = 3       -- live install output shown under the rows
S.PROVIDERS = { "Local GPU", "OpenRouter (cloud)" }
S.PROVIDER_IDS = { "local", "openrouter" }
-- mirrors server/config.py (a Python test checks they agree)
S.DEFAULTS = { provider = "local",
               local_model = "black-forest-labs/FLUX.2-klein-4B",
               openrouter_model = "google/gemini-2.5-flash-image" }
-- rows Install can fix; the API key is typed, never installed
local INSTALLABLE = { uv = true, venv = true, torch = true, deps = true,
                      model = true }

-- ---------------------------------------------------------------- pure

-- s cut to n characters; paths keep their tail, the informative end
function S.clip(s, n, keepTail)
  s = s or ""
  n = n or CLIP
  if #s <= n then return s end
  if keepTail then return "..." .. s:sub(#s - n + 4) end
  return s:sub(1, n - 3) .. "..."
end

local function indexOf(list, value)
  for i, v in ipairs(list) do if v == value then return i end end
  return nil
end

local function text(v) return type(v) == "string" and v ~= "" and v or nil end

-- The fields' first values, from the saved settings and OPENROUTER_API_KEY
function S.initial(cfg, envKey)
  cfg = cfg or {}
  local key = "API key: none saved yet"
  if text(envKey) then key = "API key: from OPENROUTER_API_KEY"
  elseif text(cfg.openrouter_api_key) then key = "API key: saved" end
  return {
    provider = indexOf(S.PROVIDER_IDS, cfg.provider) and cfg.provider
               or S.DEFAULTS.provider,
    local_model = text(cfg.local_model) or S.DEFAULTS.local_model,
    openrouter_model = text(cfg.openrouter_model) or S.DEFAULTS.openrouter_model,
    keyState = key,
  }
end

-- Every catalog model as a picker entry; planned ones are listed but marked.
-- A saved custom repo is added, so opening Setup never changes the model.
function S.localModels(catalog, current)
  local models = {}
  for _, e in ipairs(catalog or {}) do
    local planned = e.status ~= "supported"
    models[#models + 1] = {
      id = e.id, entry = e, planned = planned,
      name = e.name .. " (" .. e.params .. ")" .. (planned and " (planned)" or ""),
    }
  end
  local known = false
  for _, m in ipairs(models) do known = known or m.id == current end
  if current and not known then
    models[#models + 1] = { id = current, name = current, planned = false }
  end
  return models
end

function S.names(models)
  local out = {}
  for i, m in ipairs(models) do out[i] = m.name end
  return out
end

function S.byName(models, name)
  for _, m in ipairs(models) do if m.name == name then return m end end
  return nil
end

-- One line about what a model asks of the GPU
function S.modelInfo(model)
  if not model then return "" end
  if model.planned then return "Not supported yet." end
  local e = model.entry
  if not e then return "Custom Hugging Face repo." end
  local parts = {}
  -- %g: JSON numbers arrive as floats (8.0), shown as people write them (8)
  local full = e.vram_bf16 or (e.vram and e.vram.bf16)
  if full then
    parts[#parts + 1] = string.format("~%g GB VRAM", full)
  end
  local offload = e.vram_offload or (e.vram and e.vram.offload)
  if offload then
    parts[#parts + 1] = string.format("%g GB with offload (picked automatically)",
                                      offload)
  end
  return S.clip(table.concat(parts, ", "))
end

-- One check row as a line: "[ok] Virtual environment: .venv"
function S.itemLine(item)
  local tag
  if item.step_state == "running" then tag = "..."
  elseif item.step_state == "failed" then tag = "FAIL"
  elseif item.state == "ok" then tag = "ok"
  elseif item.state == "blocked" then tag = "!"
  else tag = "--" end
  local detail = item.detail or ""
  if item.step_state == "failed" and text(item.step_detail) then
    detail = item.step_detail
  end
  return S.clip(string.format("[%s] %s: %s", tag, item.label or item.id, detail))
end

-- Rows worth showing: the required ones for the chosen provider
function S.requiredRows(items)
  local out = {}
  for _, it in ipairs(items or {}) do
    if it.required then out[#out + 1] = it end
  end
  return out
end

-- What Install should run: required rows that are not OK yet
function S.installIds(items)
  local ids = {}
  for _, it in ipairs(items or {}) do
    if it.required and it.state ~= "ok" and INSTALLABLE[it.id] then
      ids[#ids + 1] = it.id
    end
  end
  return ids
end

-- The last n non-empty lines of a log
function S.tail(log, n)
  local lines = {}
  for line in ((log or "") .. "\n"):gmatch("(.-)\n") do
    if line:match("%S") then lines[#lines + 1] = line end
  end
  local out = {}
  for i = math.max(1, #lines - n + 1), #lines do out[#out + 1] = lines[i] end
  return out
end

-- The line under the Install button: what is left, or where to go next
function S.summary(setup)
  if setup.running then return "Installing... this can take a few minutes." end
  if setup.checking then return "Checking what is installed..." end
  local items = setup.items or {}
  for _, it in ipairs(items) do
    if it.id == "apikey" and it.required and it.state ~= "ok" then
      return "Add your OpenRouter API key above and save it."
    end
  end
  local todo = S.installIds(items)
  if #todo > 0 then return "Install adds: " .. table.concat(todo, ", ") end
  if #items == 0 then return "" end
  return "All set. Start the server from the Pixagon panel."
end

-- Has the background check settled enough to trust can_start?
function S.settled(view)
  local setup = view and view.setup
  return setup ~= nil and not setup.checking and #(setup.items or {}) > 0
end

-- ---------------------------------------------------------------- files

local function readJson(path)
  local f = io.open(path, "r")
  if not f then return nil end
  local raw = f:read("a"); f:close()
  local ok, data = pcall(json.decode, raw)
  return ok and isObject(data) and data or nil
end

-- the settings file the Python side uses (server/config.py:config_path)
local function readConfig()
  return readJson(os.getenv("PIXAGON_CONFIG")
                  or app.fs.joinPath(dataDir, "config.json"))
end

local function readCatalog()
  local osName = bootstrap.osName()
  local root = bootstrap.codeRoot(pluginDir, osName, app.fs.isFile)
  local data = root and readJson(app.fs.joinPath(
    app.fs.joinPath(root, "server"), "model_catalog.json"))
  return data and data.models or {}
end

-- ---------------------------------------------------------------- dialog

local dlg = nil
local pollTimer = nil
local viewBusy = false
local lastSetup = nil     -- the setup part of the latest view
local models = {}
local modelName = nil     -- the picker's last accepted choice

local function showProvider(isLocal)
  dlg:modify{ id = "localModel", visible = isLocal }
  dlg:modify{ id = "modelInfo", visible = isLocal }
  dlg:modify{ id = "orModel", visible = not isLocal }
  dlg:modify{ id = "apiKey", visible = not isLocal }
  dlg:modify{ id = "keyState", visible = not isLocal }
  dlg:modify{ id = "saveOr", visible = not isLocal }
end

local function note(line)
  if dlg then dlg:modify{ id = "summary", text = S.clip(line) } end
end

-- The Install part, from a supervisor view
local function paint(setup)
  local rows = S.requiredRows(setup.items)
  for i = 1, MAX_ROWS do
    local it = rows[i]
    dlg:modify{ id = "item" .. i, visible = it ~= nil,
                text = it and S.itemLine(it) or "" }
  end
  local tail = S.tail(setup.log, LOG_LINES)
  for i = 1, LOG_LINES do
    dlg:modify{ id = "log" .. i, text = S.clip(tail[i] or "") }
  end
  local todo = S.installIds(setup.items)
  dlg:modify{ id = "install",
              text = setup.running and "Installing..." or "Install",
              enabled = not setup.running and not setup.checking and #todo > 0 }
  dlg:modify{ id = "cancel", enabled = setup.running == true }
  note(S.summary(setup))
end

local function poll()
  if viewBusy or not dlg then return end
  viewBusy = true
  control.call("view", nil, function(view)
    viewBusy = false
    lastSetup = view.setup
    if dlg then paint(view.setup) end
  end, function(msg)
    viewBusy = false
    note("Not connected: " .. (msg or "?"))
  end)
end

-- Save some settings; the checks rerun for them, and the next poll shows it
local function save(values, after)
  control.call("set_model", { values = values }, function()
    if after then after() else poll() end
  end, function(msg) note("Save failed: " .. (msg or "?")) end)
end

-- The OpenRouter fields: a blank key keeps the saved one
local function saveOpenRouter(after)
  local values = { openrouter_model = dlg.data.orModel }
  if text(dlg.data.apiKey) then
    values.openrouter_api_key = dlg.data.apiKey
    dlg:modify{ id = "keyState", text = "API key: saved" }
  end
  save(values, after)
end

local function connect()
  bootstrap.ensure(note, function()
    poll()
    if Timer and not pollTimer then
      pollTimer = Timer{ interval = 1.0, ontick = poll }
      pollTimer:start()
    end
  end, function(msg) note(msg) end)
end

function S.open()
  if dlg then return end
  local init = S.initial(readConfig(), os.getenv("OPENROUTER_API_KEY"))
  models = S.localModels(readCatalog(), init.local_model)
  local isLocal = init.provider == "local"
  local current
  for _, m in ipairs(models) do
    if m.id == init.local_model then current = m end
  end
  modelName = current and current.name
  lastSetup = nil

  dlg = Dialog{ title = "Pixagon Setup", onclose = function()
    if pollTimer then pollTimer:stop(); pollTimer = nil end
    dlg = nil
  end }
  dlg:label{ text = S.clip("Data: " .. dataDir, CLIP, true) }

  dlg:separator{ text = "Backend" }
  dlg:combobox{ id = "provider", label = "Provider:", options = S.PROVIDERS,
                option = S.PROVIDERS[isLocal and 1 or 2], onchange = function()
    local i = indexOf(S.PROVIDERS, dlg.data.provider) or 1
    showProvider(i == 1)
    save({ provider = S.PROVIDER_IDS[i] })
  end }
  dlg:combobox{ id = "localModel", label = "Model:", options = S.names(models),
                option = modelName, visible = isLocal, onchange = function()
    local m = S.byName(models, dlg.data.localModel)
    if not m or m.planned then
      -- listed so the roadmap is visible, but there is no pipeline yet
      note((m and m.entry and m.entry.name or "That model") ..
           " is not supported yet.")
      dlg:modify{ id = "localModel", option = modelName }
      return
    end
    modelName = m.name
    dlg:modify{ id = "modelInfo", text = S.modelInfo(m) }
    save({ local_model = m.id })
  end }
  dlg:label{ id = "modelInfo", text = S.modelInfo(current), visible = isLocal }
  dlg:entry{ id = "orModel", label = "Model:", text = init.openrouter_model,
             visible = not isLocal }
  dlg:entry{ id = "apiKey", label = "API key:", text = "",
             visible = not isLocal }
  dlg:label{ id = "keyState", text = init.keyState, visible = not isLocal }
  dlg:button{ id = "saveOr", text = "Save OpenRouter settings",
              visible = not isLocal,
              onclick = function() saveOpenRouter() end }

  dlg:separator{ text = "Install" }
  for i = 1, MAX_ROWS do
    dlg:label{ id = "item" .. i, text = "", visible = false }
    dlg:newrow()
  end
  for i = 1, LOG_LINES do
    dlg:label{ id = "log" .. i, text = "" }
    dlg:newrow()
  end
  dlg:button{ id = "install", text = "Install", enabled = false,
              onclick = function()
    local function run()
      control.call("install", { ids = S.installIds(lastSetup and lastSetup.items) },
                   poll, function(msg) note("Install failed: " .. (msg or "?")) end)
    end
    -- typed OpenRouter fields count too, even without pressing their button
    if dlg.data.provider == S.PROVIDERS[2] then saveOpenRouter(run) else run() end
  end }
  dlg:button{ id = "cancel", text = "Cancel", enabled = false,
              onclick = function()
    control.call("cancel_install", nil, poll)
  end }
  dlg:newrow()
  dlg:label{ id = "summary", text = "Connecting to Pixagon..." }

  dlg:separator{}
  dlg:button{ text = "Close", onclick = function() dlg:close() end }
  dlg:show{ wait = false }
  connect()
end

-- The panel's Start: make sure the supervisor runs, wait for its checks,
-- then start the server, or open Setup when something is still missing.
-- onStatus(text) narrates for the panel's status line ("" when done).
function S.startServer(onStatus)
  onStatus = onStatus or function() end
  bootstrap.ensure(onStatus, function()
    local tries = 0
    local function look()
      control.call("view", nil, function(view)
        if not S.settled(view) then
          tries = tries + 1
          onStatus("Checking setup...")
          if Timer and tries < 30 then
            local t
            t = Timer{ interval = 1.0, ontick = function() t:stop(); look() end }
            t:start()
          end
          return
        end
        if view.server.running then onStatus(""); return end
        if view.server.can_start then
          onStatus("Starting the server...")
          control.call("start_server", nil, function() onStatus("") end,
                       function(msg) onStatus("Start failed: " .. (msg or "?")) end)
        else
          onStatus("Setup needs something first.")
          S.open()
        end
      end, function(msg) onStatus("Not connected: " .. (msg or "?")) end)
    end
    look()
  end, function(msg) onStatus(msg); S.open() end)
end

-- The panel's Stop
function S.stopServer(onDone)
  control.call("stop_server", nil, onDone, onDone)
end

return S
