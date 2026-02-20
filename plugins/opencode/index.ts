/**
 * Slopometry OpenCode Plugin
 *
 * Captures in-process OpenCode events and forwards them to the slopometry CLI
 * for storage, analysis, and feedback injection.
 *
 * Architecture:
 * - Hooks into OpenCode's plugin system (tool.execute.before/after, bus events, system transform)
 * - Spawns `slopometry hook-opencode --event-type <type>` per event with JSON on stdin
 * - Reads stdout for feedback (code smells, coverage warnings)
 * - Injects feedback via tool output mutation or system prompt transform
 *
 * Registration: Add to opencode.json:
 *   "plugin": ["file:///path/to/slopometry/plugins/opencode"]
 *
 * Or install via: slopometry install --target opencode
 */

import type { Plugin, Hooks, PluginInput } from "@opencode-ai/plugin"
import type { Event } from "@opencode-ai/sdk"

// --- Types ---

interface ForwardResult {
  stdout: string
  exitCode: number
}

interface ToolStartRecord {
  startTime: number
  tool: string
  args: any
}

// --- Core forwarding ---

/**
 * Spawn slopometry CLI and forward event data on stdin.
 * Returns stdout (feedback) and exit code.
 */
async function forward(eventType: string, payload: object): Promise<ForwardResult> {
  try {
    const proc = Bun.spawn(["slopometry", "hook-opencode", "--event-type", eventType], {
      stdin: new Blob([JSON.stringify(payload)]),
      stdout: "pipe",
      stderr: "pipe",
    })
    const stdout = await new Response(proc.stdout).text()
    const exitCode = await proc.exited
    return { stdout: stdout.trim(), exitCode }
  } catch (e) {
    // slopometry not installed or not in PATH - fail silently
    return { stdout: "", exitCode: 1 }
  }
}

/**
 * Parse slopometry feedback from stdout.
 *
 * Slopometry outputs JSON with {decision, reason} for blocking feedback,
 * or plain text for informational messages.
 */
function parseFeedback(stdout: string): string | null {
  if (!stdout) return null
  try {
    const parsed = JSON.parse(stdout)
    if (parsed.reason) return parsed.reason
    if (parsed.decision === "block" && parsed.reason) return parsed.reason
    return null
  } catch {
    // Plain text feedback
    return stdout || null
  }
}

// --- Plugin Export ---

export const SlopometryPlugin: Plugin = async (ctx: PluginInput) => {
  // Per-session state
  const toolStarts = new Map<string, ToolStartRecord>()
  let pendingFeedback: string | null = null
  const sessionStartTimes = new Map<string, number>()

  // Fetch OpenCode version once at plugin init
  let opencodeVersion: string | undefined
  try {
    const health = await ctx.client.global.health()
    opencodeVersion = (health.data as any)?.version
  } catch {
    // Server may not support health endpoint - leave undefined
  }

  const hooks: Hooks = {
    // -------------------------------------------------------
    // Bus event listener - capture all internal bus events
    // -------------------------------------------------------
    event: async ({ event }: { event: Event }) => {
      const type = event.type
      const props = (event as any).properties

      switch (type) {
        case "session.created": {
          const info = props?.info
          if (!info) break
          sessionStartTimes.set(info.id, Date.now())

          // Track subagent sessions
          if (info.parentID) {
            await forward("subagent_start", {
              session_id: info.id,
              parent_id: info.parentID,
              title: info.title,
            })
          }
          break
        }

        case "session.idle": {
          const sessionID = props?.sessionID
          if (!sessionID) break

          // Fetch transcript via SDK for the stop event
          let transcript: any[] | undefined
          let sessionTokens: any = undefined
          let sessionCost = 0
          let todos: any[] = []

          try {
            const response = await ctx.client.session.messages({
              path: { id: sessionID },
            })
            if (response.data) {
              transcript = (response.data as any[]).map((msg: any) => ({
                role: msg.info?.role,
                tokens: msg.info?.tokens,
                cost: msg.info?.cost,
                modelID: msg.info?.modelID,
                agent: msg.info?.agent,
                parts: msg.parts?.map((p: any) => ({
                  type: p.type,
                  tool: p.tool,
                  text: p.type === "text" ? p.text?.slice(0, 500) : undefined,
                  state: p.state
                    ? {
                        status: p.state.status,
                        title: p.state.title,
                        time: p.state.time,
                      }
                    : undefined,
                })),
              }))

              // Aggregate tokens and cost
              for (const msg of response.data as any[]) {
                if (msg.info?.role === "assistant" && msg.info?.tokens) {
                  sessionCost += msg.info.cost || 0
                  if (!sessionTokens) {
                    sessionTokens = {
                      input: 0,
                      output: 0,
                      reasoning: 0,
                      cache_read: 0,
                      cache_write: 0,
                    }
                  }
                  sessionTokens.input += msg.info.tokens.input || 0
                  sessionTokens.output += msg.info.tokens.output || 0
                  sessionTokens.reasoning += msg.info.tokens.reasoning || 0
                  sessionTokens.cache_read += msg.info.tokens.cache?.read || 0
                  sessionTokens.cache_write += msg.info.tokens.cache?.write || 0
                }
              }
            }
          } catch {
            // SDK call failed - send stop event without transcript
          }

          const result = await forward("stop", {
            session_id: sessionID,
            tokens: sessionTokens,
            cost: sessionCost,
            todos,
            transcript,
            opencode_version: opencodeVersion,
          })

          // Cache any stop feedback for system prompt injection
          const feedback = parseFeedback(result.stdout)
          if (feedback) {
            pendingFeedback = feedback
          }
          break
        }

        case "message.updated": {
          const info = props?.info
          if (!info || info.role !== "assistant") break

          await forward("message_updated", {
            session_id: info.sessionID,
            message_id: info.id,
            model_id: info.modelID,
            provider_id: info.providerID,
            agent: info.agent,
            tokens: {
              input: info.tokens?.input || 0,
              output: info.tokens?.output || 0,
              reasoning: info.tokens?.reasoning || 0,
              cache_read: info.tokens?.cache?.read || 0,
              cache_write: info.tokens?.cache?.write || 0,
            },
            cost: info.cost || 0,
          })
          break
        }

        case "todo.updated": {
          if (!props?.sessionID || !props?.todos) break

          await forward("todo_updated", {
            session_id: props.sessionID,
            todos: props.todos.map((t: any) => ({
              content: t.content,
              status: t.status,
              priority: t.priority,
            })),
          })
          break
        }
      }
    },

    // -------------------------------------------------------
    // Tool execution hooks
    // -------------------------------------------------------
    "tool.execute.before": async (input, output) => {
      const key = `${input.sessionID}:${input.callID}`
      toolStarts.set(key, {
        startTime: Date.now(),
        tool: input.tool,
        args: output.args,
      })

      await forward("pre_tool_use", {
        session_id: input.sessionID,
        call_id: input.callID,
        tool: input.tool,
        args: output.args,
      })
    },

    "tool.execute.after": async (input, output) => {
      const key = `${input.sessionID}:${input.callID}`
      const startRecord = toolStarts.get(key)
      const durationMs = startRecord ? Date.now() - startRecord.startTime : undefined
      toolStarts.delete(key)

      const result = await forward("post_tool_use", {
        session_id: input.sessionID,
        call_id: input.callID,
        tool: input.tool,
        args: input.args,
        output: output.output?.slice(0, 10000), // Truncate large outputs
        duration_ms: durationMs,
        title: output.title,
      })

      // Inject inline feedback from slopometry
      const feedback = parseFeedback(result.stdout)
      if (feedback) {
        output.output += `\n\n---\n[slopometry] ${feedback}`
      }
    },

    // -------------------------------------------------------
    // System prompt injection for cached feedback
    // -------------------------------------------------------
    "experimental.chat.system.transform": async (_input, output) => {
      if (pendingFeedback) {
        output.system.push(
          `<slopometry-feedback>\n${pendingFeedback}\n</slopometry-feedback>`,
        )
        pendingFeedback = null
      }
    },
  }

  return hooks
}

// Default export for OpenCode plugin loader
export default SlopometryPlugin
