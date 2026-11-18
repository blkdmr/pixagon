-- Animation dialog: generate frame sequences via the animate mode,
-- preview them, and insert into Aseprite's timeline.
local pluginDir = ...
local function moduleFrom(name)
  return assert(loadfile(app.fs.joinPath(pluginDir, name)))(pluginDir)
end

local client = moduleFrom("client.lua")
local P = moduleFrom("prompt.lua")
local ui = moduleFrom("ui.lua")
local sprite = moduleFrom("sprite.lua")

local A = {}

-- Persistent settings across reopenings.
local last = { preset = "Walk cycle", prompt = "", frames = 8,
               strength = 55, fps = 8, loop = true }

local PRESET_OPTS = P.ANIM_PRESET_ORDER
local FPS_OPTS = { "6", "8", "12", "15", "24" }

local STATUS_W, STATUS_H = 282, 48

-- ---- Preview dialog (shown after generation, before timeline insertion)

local function showPreview(frameImages, durations, loop, onInsert, onDiscard)
  local current = 1
  local playing = false
  local playTimer

  local pw = math.min(282, math.max(64, frameImages[1].width * 4))
  local ph = math.min(282, math.max(64, frameImages[1].height * 4))

  local dlg = Dialog{ title = "Pixagon - Animation Preview", resizeable = false }

  dlg:canvas{
    id = "preview", width = pw, height = ph,
    onpaint = function(ev)
      local gc = ev.context
      ui.drawChecker(gc, 0, 0, pw, ph)
      local img = frameImages[current]
      local s = math.min(pw / img.width, ph / img.height)
      if s >= 1 then s = math.floor(s) end
      local dw = math.max(1, math.floor(img.width * s))
      local dh = math.max(1, math.floor(img.height * s))
      local dx = (pw - dw) // 2
      local dy = (ph - dh) // 2
      gc:drawImage(img, Rectangle(0, 0, img.width, img.height),
                   Rectangle(dx, dy, dw, dh))
    end,
  }
  dlg:label{ id = "framelabel",
             text = string.format("Frame %d/%d", current, #frameImages) }
  dlg:button{ id = "playbtn", text = "Play", onclick = function()
    playing = not playing
    dlg:modify{ id = "playbtn", text = playing and "Pause" or "Play" }
    if playing and Timer then
      local ms = durations[1] or 125
      playTimer = Timer{ interval = ms / 1000.0, ontick = function()
        current = current % #frameImages + 1
        dlg:modify{ id = "framelabel",
                    text = string.format("Frame %d/%d", current, #frameImages) }
        dlg:repaint()
      end }
      playTimer:start()
    elseif playTimer then
      playTimer:stop()
      playTimer = nil
    end
  end }
  dlg:button{ text = "Insert", onclick = function()
    if playTimer then playTimer:stop() end
    dlg:close()
    onInsert()
  end }
  dlg:button{ text = "Discard", onclick = function()
    if playTimer then playTimer:stop() end
    dlg:close()
    onDiscard()
  end }

  dlg:show{
    wait = true,
    onclose = function()
      if playTimer then playTimer:stop(); playTimer = nil end
    end,
  }
end

-- ---- Main animation dialog

function A.open()
  local state = "idle"
  local statusText = "Set the parameters and press Run."
  local progress = 0
  local job = nil
  local animTimer

  local dlg

  local function repaint() dlg:repaint() end

  local function setState(s, text)
    state = s
    statusText = text
    dlg:modify{ id = "run", enabled = s ~= "running" }
    dlg:modify{ id = "cancel", enabled = s == "running" }
    if Timer then
      if s == "running" and not animTimer then
        animTimer = Timer{ interval = 0.08, ontick = repaint }
        animTimer:start()
      elseif s ~= "running" and animTimer then
        animTimer:stop()
        animTimer = nil
      end
    end
    repaint()
  end

  local function presetChanged()
    local d = dlg.data
    local key = P.ANIM_PRESETS[d.preset]
    if key then
      -- fill defaults from preset
      local defaults = P.ANIM_PRESET_DEFAULTS[key]
      if defaults then
        dlg:modify{ id = "frames", value = defaults.frames }
        dlg:modify{ id = "strength", value = math.floor(defaults.strength * 100) }
        dlg:modify{ id = "loop", selected = defaults.loop }
      end
    end
  end

  local function startRun()
    if state == "running" then return end
    local d = dlg.data
    last.preset = d.preset
    last.prompt = d.prompt
    last.frames = d.frames
    last.strength = d.strength
    last.fps = tonumber(d.fps) or 8
    last.loop = d.loop

    if d.prompt == "" then
      setState("error", "Prompt describes the subject.")
      return
    end

    local spr = app.sprite
    local payload = {
      id = sprite.newId(),
      mode = "animate",
      prompt = d.prompt,
      target_size = { spr and spr.width or 64, spr and spr.height or 64 },
      variants = 1,
      frames = {},
      animation = {
        frame_count = d.frames,
        strength = d.strength / 100.0,
        loop = d.loop,
        fps = last.fps,
        preset = P.ANIM_PRESETS[d.preset],
      },
    }

    -- if a sprite is open, use current frame as starting frame
    if spr then
      payload.frames = { { image = sprite.exportFrame() } }
    end

    progress = 0
    setState("running", "Contacting server...")

    job = client.request(payload, {
      onprogress = function(v, stage, frameIdx, frameTotal)
        if stage then
          statusText = stage
        else
          progress = v
          if frameIdx and frameTotal then
            statusText = string.format("Frame %d/%d", frameIdx + 1, frameTotal)
          else
            statusText = "Generating"
          end
        end
        repaint()
      end,
      onresult = function(images, seeds)
        -- should not arrive for animate mode, but handle gracefully
        setState("done", "Done (unexpected result type).")
      end,
      onanimresult = function(frameImgs, frameDurations, seeds, isLoop)
        local imgs = {}
        for n, s in ipairs(frameImgs) do
          imgs[n] = sprite.imageFromPayload(s, n)
        end
        setState("done", string.format(
          "%d frames ready.", #imgs))

        showPreview(imgs, frameDurations, isLoop, function()
          -- Insert into timeline
          sprite.insertAnimation(imgs, frameDurations, "Pixagon Anim")
          setState("done", "Animation inserted into timeline.")
        end, function()
          setState("idle", "Discarded. Press Run to try again.")
        end)
      end,
      onerror = function(msg)
        setState("error", msg)
      end,
    })
  end

  local function paintStatus(ev)
    local gc = ev.context
    local face = ui.face()
    ui.drawInsetPanel(gc, 0, 0, STATUS_W, STATUS_H)
    if state == "error" then
      gc.color = Color{ r = 168, g = 82, b = 62 }
    else
      gc.color = ui.text()
    end
    local line = statusText
    if state == "running" then
      line = line:gsub("%.+$", "")
      line = line .. string.rep(".", math.floor(os.clock() * 3) % 4)
    end
    gc:fillText(line, 10, 8)

    if state == "running" and progress > 0 then
      local bx, by, bw, bh = 8, 28, STATUS_W - 16, 12
      gc.color = ui.shade(face, 0.82)
      gc:fillRect(Rectangle(bx, by, bw, bh))
      gc.color = ui.shade(face, 0.50)
      gc:fillRect(Rectangle(bx + 1, by + 1,
                            math.floor((bw - 2) * progress), bh - 2))
    end
  end

  dlg = Dialog{
    title = "Pixagon - Animate",
    resizeable = false,
    onclose = function()
      if animTimer then animTimer:stop() end
      if job and state == "running" then job.cancel() end
    end,
  }

  dlg:separator{ text = "Animation" }
  dlg:combobox{ id = "preset", label = "Preset:", option = last.preset,
                options = PRESET_OPTS, onchange = presetChanged }
  dlg:entry{ id = "prompt", label = "Subject:", text = last.prompt, focus = true }
  dlg:slider{ id = "frames", label = "Frames:", min = 2, max = 32,
              value = last.frames }
  dlg:slider{ id = "strength", label = "Strength:", min = 10, max = 90,
              value = last.strength }
  dlg:combobox{ id = "fps", label = "FPS:", option = tostring(last.fps),
                options = FPS_OPTS }
  dlg:check{ id = "loop", text = "Loop (seamless)", selected = last.loop }
  dlg:separator{ text = "Status" }
  dlg:canvas{ id = "view", width = STATUS_W, height = STATUS_H,
              onpaint = paintStatus }
  dlg:button{ id = "run", text = "Run", onclick = startRun }
  dlg:button{ id = "cancel", text = "Cancel", enabled = false,
              onclick = function()
    if job then job.cancel() end
    setState("idle", "Cancelled. Press Run to try again.")
  end }
  dlg:button{ text = "Close", onclick = function() dlg:close() end }

  dlg:show{ wait = false }
end

return A
