-- Original work Copyright 2026 Valentyn Karachevtsev (SpriteLoom)
-- Licensed under the Apache License, Version 2.0
-- Modifications Copyright 2026 Alessio Russo (Pixagon)

local services, heartbeat

-- The supervisor exits after 180 s without a request; this keeps it alive
-- while Aseprite runs, so Aseprite crashing is the only way to go quiet
local HEARTBEAT_SECONDS = 60

function init(plugin)
  local pluginDir = plugin.path
  local function load(name, ...)
    return assert(loadfile(app.fs.joinPath(pluginDir, name)))(...)
  end

  -- one instance of each, shared by the panel and Setup
  local bootstrap = load("bootstrap.lua", pluginDir)
  local dataDir = bootstrap.dataDir(bootstrap.osName())
  local control = load("control.lua", dataDir)
  bootstrap.control = control
  local setup = load("setup.lua", pluginDir, bootstrap, control, dataDir)
  services = { bootstrap = bootstrap, control = control, setup = setup,
               dataDir = dataDir }
  local dialogs = load("dialogs.lua", pluginDir, services)

  plugin:newCommand{
    id = "PixagonOpen",
    title = "Pixagon...",
    group = "sprite_properties",
    onclick = function() dialogs.open() end,
  }
  plugin:newCommand{
    id = "PixagonSetup",
    title = "Pixagon Setup...",
    group = "sprite_properties",
    onclick = function() setup.open() end,
  }

  if Timer then
    heartbeat = Timer{ interval = HEARTBEAT_SECONDS, ontick = function()
      -- only once a supervisor exists; pinging never starts one
      if control.readState() then
        control.ping(function() end, function() end)
      end
    end }
    heartbeat:start()
  end
end

-- Aseprite is closing (or the extension is being disabled/updated): stop
-- the supervisor, which stops the model server with it
function exit(_)
  if heartbeat then heartbeat:stop() end
  if services then services.control.shutdown() end
end
