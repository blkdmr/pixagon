-- Plain-Lua tests for control.lua against a fake WebSocket: run with
-- `lua5.4 plugin/tests/test_control.lua` from the repo root.
local failures = 0
local function eq(got, want, what)
  if got ~= want then
    failures = failures + 1
    print(string.format("FAIL %s\n  got:  %s\n  want: %s",
                        what, tostring(got), tostring(want)))
  end
end

-- server.json the control client reads; nil = no supervisor
local STATE = nil
local io_open = io.open
io.open = function(path, mode)
  if path:match("server%.json$") then
    if not STATE then return nil end
    return { read = function() return STATE end, close = function() end }
  end
  return io_open(path, mode)
end

-- json stand-in: requests are tables we inspect directly, replies are
-- tables handed back by decode
local lastDecoded = nil
json = {
  encode = function(t) return t end,
  decode = function(text)
    if text == STATE then return { control = 8790, port = 8765 } end
    return lastDecoded
  end,
}
app = { fs = { joinPath = function(a, b) return a .. "/" .. b end } }
Timer = nil
WebSocketMessageType = { OPEN = 0, TEXT = 1, CLOSE = 2 }

local sockets = {}
WebSocket = function(t)
  local ws = { url = t.url, sent = {}, closed = false, on = t.onreceive }
  ws.connect = function() end
  ws.close = function() ws.closed = true end
  ws.sendText = function(_, msg) ws.sent[#ws.sent + 1] = msg end
  sockets[#sockets + 1] = ws
  return ws
end

local function reply(ws, msg)
  lastDecoded = msg
  ws.on(WebSocketMessageType.TEXT, "x")
end

local C = assert(loadfile("plugin/control.lua"))("/data")

-- no server.json: the call fails at once, no socket opened
local err
C.call("view", nil, function() end, function(m) err = m end)
eq(err, "Pixagon is not running", "no supervisor")
eq(#sockets, 0, "no socket without server.json")

-- calls queue while connecting and go out on OPEN, in order
STATE = "state"
local got = {}
C.call("view", nil, function(r) got.view = r end)
C.call("recheck", nil, function(r) got.recheck = r end)
eq(#sockets, 1, "one socket for both calls")
local ws = sockets[1]
eq(ws.url, "http://127.0.0.1:8790", "control port from server.json")
eq(#ws.sent, 0, "nothing sent before OPEN")
ws.on(WebSocketMessageType.OPEN, "")
eq(C.isOpen(), true, "open after OPEN")
eq(#ws.sent, 2, "queue flushed on OPEN")
eq(ws.sent[1].method, "view", "first call first")

-- replies match by id, not by order
reply(ws, { id = ws.sent[2].id, type = "reply", result = "R" })
reply(ws, { id = ws.sent[1].id, type = "reply", result = "V" })
eq(got.view, "V", "view reply routed by id")
eq(got.recheck, "R", "recheck reply routed by id")

-- an error reply goes to onError
local why
C.call("install", { ids = {} }, nil, function(m) why = m end)
reply(ws, { id = ws.sent[3].id, type = "error", message = "disk full" })
eq(why, "disk full", "error reply")

-- a CLOSE fails whatever is still waiting
local lost
C.call("view", nil, nil, function(m) lost = m end)
ws.on(WebSocketMessageType.CLOSE, "")
eq(lost, "Pixagon stopped", "pending call failed on close")
eq(C.isOpen(), false, "idle after close")

-- the next call reconnects
C.call("ping", nil, function() end)
eq(#sockets, 2, "reconnects on demand")

-- shutdown sends on an open socket and closes it
sockets[2].on(WebSocketMessageType.OPEN, "")
C.shutdown()
eq(sockets[2].sent[#sockets[2].sent].method, "shutdown", "shutdown sent")
eq(sockets[2].closed, true, "socket closed after shutdown")

io.open = io_open
if failures > 0 then
  print(failures .. " failure(s)")
  os.exit(1)
end
print("test_control ok")
