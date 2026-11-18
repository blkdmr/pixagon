-- Talks to the supervisor (launcher.daemon) over one WebSocket: each call
-- gets an id, and the reply with that id runs its callback. The port comes
-- from server.json in the data folder, re-read on every new connection.
local dataDir = ...

-- Aseprite's json.decode returns userdata (indexable, with # and ipairs),
-- not a table; plain-Lua tests return tables. Either is a decoded object.
local function isObject(v)
  return type(v) == "table" or type(v) == "userdata"
end

local C = {}
local CONNECT_TIMEOUT = 2.0

local ws = nil
local state = "idle"      -- idle | connecting | open
local queue = {}          -- calls waiting for the socket to open
local pending = {}        -- id -> callbacks, sent and awaiting a reply
local nextId = 1
local connectTimer = nil

-- server.json as a table ({control, port, pid}), or nil
function C.readState()
  local f = io.open(app.fs.joinPath(dataDir, "server.json"), "r")
  if not f then return nil end
  local text = f:read("a"); f:close()
  local ok, data = pcall(json.decode, text)
  if ok and isObject(data) and type(data.control) == "number" then
    return data
  end
  return nil
end

function C.isOpen() return state == "open" end

-- Fail every waiting call with msg and forget the socket
local function drop(msg)
  if connectTimer then connectTimer:stop(); connectTimer = nil end
  if ws then pcall(function() ws:close() end) end
  ws, state = nil, "idle"
  local failed = {}
  for _, cb in pairs(pending) do failed[#failed + 1] = cb end
  for _, item in ipairs(queue) do failed[#failed + 1] = item end
  pending, queue = {}, {}
  for _, cb in ipairs(failed) do
    if cb.onError then cb.onError(msg) end
  end
end

-- Send everything queued while the socket was connecting
local function flush()
  local items = queue
  queue = {}
  for _, item in ipairs(items) do
    pending[item.msg.id] = item
    local ok = pcall(function() ws:sendText(json.encode(item.msg)) end)
    if not ok then drop("Pixagon stopped"); return end
  end
end

local function connect()
  local info = C.readState()
  if not info then drop("Pixagon is not running"); return end
  state = "connecting"
  local socket
  socket = WebSocket{
    url = "http://127.0.0.1:" .. math.floor(info.control),
    deflate = false,
    onreceive = function(mt, data)
      if socket ~= ws then return end  -- a socket we already dropped
      if mt == WebSocketMessageType.OPEN then
        if connectTimer then connectTimer:stop(); connectTimer = nil end
        state = "open"
        flush()
      elseif mt == WebSocketMessageType.TEXT then
        local ok, msg = pcall(json.decode, data)
        if not ok or not isObject(msg) then return end
        local cb = pending[msg.id]
        pending[msg.id] = nil
        if not cb then return end
        if msg.type == "reply" then
          if cb.onReply then cb.onReply(msg.result) end
        elseif cb.onError then
          cb.onError(msg.message or "error")
        end
      elseif mt == WebSocketMessageType.CLOSE then
        drop("Pixagon stopped")
      end
    end,
  }
  ws = socket
  if Timer then
    -- a dead port never fires CLOSE; give up on it after a moment
    connectTimer = Timer{ interval = CONNECT_TIMEOUT, ontick = function()
      if state == "connecting" then drop("Pixagon is not running") end
    end }
    connectTimer:start()
  end
  socket:connect()
end

-- Call a supervisor method; onReply(result) or onError(message)
function C.call(method, params, onReply, onError)
  local id = nextId
  nextId = nextId + 1
  queue[#queue + 1] = { msg = { id = id, method = method, params = params },
                        onReply = onReply, onError = onError }
  if state == "open" then flush()
  elseif state == "idle" then connect() end
end

function C.ping(onOk, onFail)
  C.call("ping", nil, onOk, onFail)
end

-- Ask the supervisor to stop it all (Aseprite is closing); fire and forget
function C.shutdown()
  if state == "open" and ws then
    pcall(function()
      ws:sendText(json.encode({ id = 0, method = "shutdown" }))
    end)
  end
  drop(nil)
end

return C
