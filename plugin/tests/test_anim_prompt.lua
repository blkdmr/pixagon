-- Plain-Lua tests for animation presets in prompt.lua.
-- Run with: lua5.4 plugin/tests/test_anim_prompt.lua
local P = dofile("plugin/prompt.lua")

local failures = 0
local function eq(got, want, what)
  if got ~= want then
    failures = failures + 1
    print(string.format("FAIL %s\n  got:  %s\n  want: %s",
                        what, tostring(got), tostring(want)))
  end
end

local function ok(cond, what)
  if not cond then
    failures = failures + 1
    print("FAIL " .. what)
  end
end

-- Every ordered preset must have an entry in the lookup table.
for _, label in ipairs(P.ANIM_PRESET_ORDER) do
  ok(P.ANIM_PRESETS[label] ~= nil or label == "Custom",
     "anim preset order has entry: " .. label)
end

-- Protocol keys map to known server presets (or nil for Custom).
eq(P.ANIM_PRESETS["Walk cycle"], "walk_cycle", "walk_cycle key")
eq(P.ANIM_PRESETS["Idle"], "idle", "idle key")
eq(P.ANIM_PRESETS["Attack"], "attack", "attack key")
eq(P.ANIM_PRESETS["Custom"], nil, "custom key is nil")

-- Default values exist for every non-custom preset.
for _, label in ipairs(P.ANIM_PRESET_ORDER) do
  local key = P.ANIM_PRESETS[label]
  if key then
    local d = P.ANIM_PRESET_DEFAULTS[key]
    ok(d ~= nil, "defaults exist for " .. key)
    ok(type(d.frames) == "number" and d.frames >= 2, "frames for " .. key)
    ok(type(d.strength) == "number" and d.strength > 0, "strength for " .. key)
    ok(type(d.loop) == "boolean", "loop for " .. key)
  end
end

-- Custom has no defaults entry.
eq(P.ANIM_PRESET_DEFAULTS[nil], nil, "no defaults for nil key")

if failures == 0 then
  print("anim_prompt.lua: all tests passed")
else
  print(failures .. " failure(s)")
  os.exit(1)
end
