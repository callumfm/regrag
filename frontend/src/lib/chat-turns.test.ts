import { describe, expect, it } from "vitest"
import { type ChatAction, type ChatTurn, chatReducer } from "./chat-turns"

function asked(): ChatTurn[] {
	return chatReducer([], { type: "ask", id: "t1", question: "q" })
}

function run(...actions: ChatAction[]): ChatTurn {
	return actions.reduce(chatReducer, asked())[0]
}

const retrieve: ChatAction = {
	event: "step",
	data: { step: "retrieve", ms: 12 },
}
const assess: ChatAction = { event: "step", data: { step: "assess", ms: 90 } }

describe("chatReducer", () => {
	it("collects the steps a run reports, in the order they arrive", () => {
		const turn = run(retrieve, assess)

		expect(turn.steps.map((step) => step.step)).toEqual(["retrieve", "assess"])
	})

	it("counts a step as the run being under way", () => {
		expect(run(retrieve).status).toBe("streaming")
	})

	it("keeps the steps of a turn that failed", () => {
		const turn = run(retrieve, {
			event: "error",
			data: { error: "LLMError", message: "boom", request_id: null },
		})

		expect(turn.status).toBe("failed")
		expect(turn.steps).toHaveLength(1)
	})

	it("starts a turn with no steps", () => {
		expect(asked()[0].steps).toEqual([])
	})
})
