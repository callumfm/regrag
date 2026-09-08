import { describe, expect, it } from "vitest"
import { type ChatAction, type ChatTurn, chatReducer } from "./chat-turns"

function asked(): ChatTurn[] {
	return chatReducer([], { type: "ask", id: "t1", question: "q" })
}

function run(...actions: ChatAction[]): ChatTurn {
	return actions.reduce(chatReducer, asked())[0]
}

function started(step: string, subject: string | null = null): ChatAction {
	return {
		event: "step",
		data: { step, ms: 0, status: "running", subject },
	} as ChatAction
}

function finished(
	step: string,
	ms: number,
	subject: string | null = null,
): ChatAction {
	return {
		event: "step",
		data: { step, ms, status: "completed", subject },
	} as ChatAction
}

describe("chatReducer", () => {
	it("adds a step to the trail when it starts", () => {
		const turn = run(started("retrieve"))

		expect(turn.steps.map((step) => [step.step, step.status])).toEqual([
			["retrieve", "running"],
		])
	})

	it("settles a running step in place rather than adding it twice", () => {
		const turn = run(started("retrieve"), finished("retrieve", 120))

		expect(turn.steps).toHaveLength(1)
		expect(turn.steps[0]).toMatchObject({ status: "completed", ms: 120 })
	})

	it("settles concurrent calls in the order they started", () => {
		const turn = run(
			started("tool_search", "penalties"),
			started("tool_search", "verification"),
			finished("tool_search", 500, "penalties"),
		)

		expect(turn.steps.map((step) => [step.subject, step.status])).toEqual([
			["penalties", "completed"],
			["verification", "running"],
		])
	})

	it("keeps the trail in the order the run walked it", () => {
		const turn = run(
			started("retrieve"),
			finished("retrieve", 12),
			started("assess"),
			finished("assess", 90),
		)

		expect(turn.steps.map((step) => step.step)).toEqual(["retrieve", "assess"])
	})

	it("counts a started step as the run being under way", () => {
		expect(run(started("retrieve")).status).toBe("streaming")
	})

	it("keeps an unfinished step when the turn fails", () => {
		const turn = run(started("retrieve"), {
			event: "error",
			data: { error: "LLMError", message: "boom", request_id: null },
		})

		expect(turn.status).toBe("failed")
		expect(turn.steps).toMatchObject([{ status: "running" }])
	})

	it("adds a finished step that was never announced as starting", () => {
		const turn = run(finished("retrieve", 12))

		expect(turn.steps).toMatchObject([
			{ step: "retrieve", status: "completed" },
		])
	})

	it("starts a turn with no steps", () => {
		expect(asked()[0].steps).toEqual([])
	})
})
