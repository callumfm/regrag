import type { Element, Root } from "hast"
import { describe, expect, it } from "vitest"
import type { ChatSource } from "@/api/types"
import {
	citedSources,
	extractCitedMarkers,
	moveMarkersAfterPunctuation,
	numberCitations,
	rehypeCitationMarkers,
	renumberCitations,
	splitCitationMarkers,
} from "./citations"

const known = new Set([1, 2, 7, 12])

describe("moveMarkersAfterPunctuation", () => {
	it("moves a marker past the punctuation it precedes", () => {
		expect(moveMarkersAfterPunctuation("claim [1].", known)).toBe("claim.[1]")
	})

	it("keeps a run of markers together", () => {
		expect(moveMarkersAfterPunctuation("claim [1][2].", known)).toBe(
			"claim.[1][2]",
		)
	})

	it("moves markers that are not preceded by a space", () => {
		expect(moveMarkersAfterPunctuation("claim[1],", known)).toBe("claim,[1]")
	})

	it("handles every punctuation mark it recognises", () => {
		expect(
			moveMarkersAfterPunctuation("a [1]. b [1], c [1]; d [1]:", known),
		).toBe("a.[1] b,[1] c;[1] d:[1]")
	})

	it("leaves a run alone when any marker is unknown", () => {
		expect(moveMarkersAfterPunctuation("claim [1][9].", known)).toBe(
			"claim [1][9].",
		)
	})

	it("leaves markers alone when no punctuation follows", () => {
		expect(moveMarkersAfterPunctuation("claim [1] and more", known)).toBe(
			"claim [1] and more",
		)
	})
})

describe("splitCitationMarkers", () => {
	it("splits text around a known marker", () => {
		expect(splitCitationMarkers("a [1] b", known)).toEqual([
			{ kind: "text", value: "a " },
			{ kind: "marker", marker: 1 },
			{ kind: "text", value: " b" },
		])
	})

	it("applies the punctuation move before splitting", () => {
		expect(splitCitationMarkers("claim [1].", known)).toEqual([
			{ kind: "text", value: "claim." },
			{ kind: "marker", marker: 1 },
		])
	})

	it("emits adjacent markers as separate segments", () => {
		expect(splitCitationMarkers("claim [1][2].", known)).toEqual([
			{ kind: "text", value: "claim." },
			{ kind: "marker", marker: 1 },
			{ kind: "marker", marker: 2 },
		])
	})

	it("leaves unknown markers as literal text", () => {
		expect(splitCitationMarkers("a [9] b", known)).toEqual([
			{ kind: "text", value: "a [9] b" },
		])
	})

	it("returns a single text segment when nothing is cited", () => {
		expect(splitCitationMarkers("no citations here", known)).toEqual([
			{ kind: "text", value: "no citations here" },
		])
	})

	it("returns nothing for an empty string", () => {
		expect(splitCitationMarkers("", known)).toEqual([])
	})
})

describe("extractCitedMarkers", () => {
	it("returns known markers in order of first appearance", () => {
		expect(extractCitedMarkers("[7] then [2] then [12]", known)).toEqual([
			7, 2, 12,
		])
	})

	it("reports a repeated marker once", () => {
		expect(extractCitedMarkers("[2] and [1] and [2]", known)).toEqual([2, 1])
	})

	it("ignores markers outside the known set", () => {
		expect(extractCitedMarkers("[9] and [1]", known)).toEqual([1])
	})

	it("ignores markers inside inline code", () => {
		expect(extractCitedMarkers("read `arr[1]` then cite [7]", known)).toEqual([
			7,
		])
	})

	it("ignores markers inside a fenced block", () => {
		expect(
			extractCitedMarkers("cite [7]\n```ts\nconst x = arr[1]\n```\nend", known),
		).toEqual([7])
	})

	it("ignores markers inside a tilde fence", () => {
		expect(extractCitedMarkers("cite [7]\n~~~\narr[1]\n~~~", known)).toEqual([
			7,
		])
	})

	it("ignores markers inside a fence that is still streaming", () => {
		expect(
			extractCitedMarkers("cite [7]\n```ts\nconst x = arr[1]", known),
		).toEqual([7])
	})

	it("keeps markers that share a line with inline code", () => {
		expect(extractCitedMarkers("`arr[1]` supports [2]", known)).toEqual([2])
	})
})

describe("numberCitations", () => {
	it("renumbers retrieval positions as 1..k", () => {
		expect([...numberCitations("[7] and [12]", known)]).toEqual([
			[7, 1],
			[12, 2],
		])
	})

	it("is empty when the answer cites nothing known", () => {
		expect(numberCitations("[9] only", known).size).toBe(0)
	})

	it("does not let a marker inside code consume a display number", () => {
		expect([...numberCitations("`arr[1]` then [7] and [12]", known)]).toEqual([
			[7, 1],
			[12, 2],
		])
	})
})

function paragraph(...values: string[]): Root {
	return {
		type: "root",
		children: [
			{
				type: "element",
				tagName: "p",
				properties: {},
				children: values.map((value) => ({ type: "text", value })),
			},
		],
	}
}

function firstChild(tree: Root): Element {
	return tree.children[0] as Element
}

describe("rehypeCitationMarkers", () => {
	it("replaces known markers with cite-marker elements", () => {
		const tree = paragraph("claim [1].")
		rehypeCitationMarkers(known)()(tree)
		expect(firstChild(tree).children).toEqual([
			{ type: "text", value: "claim." },
			{
				type: "element",
				tagName: "cite-marker",
				properties: { marker: "1" },
				children: [],
			},
		])
	})

	it("leaves text without known markers untouched", () => {
		const tree = paragraph("claim [9].")
		rehypeCitationMarkers(known)()(tree)
		expect(firstChild(tree).children).toEqual([
			{ type: "text", value: "claim [9]." },
		])
	})

	it("skips markers inside code elements", () => {
		const tree: Root = {
			type: "root",
			children: [
				{
					type: "element",
					tagName: "code",
					properties: {},
					children: [{ type: "text", value: "arr[1]" }],
				},
			],
		}
		rehypeCitationMarkers(known)()(tree)
		expect(firstChild(tree).children).toEqual([
			{ type: "text", value: "arr[1]" },
		])
	})

	it("rewrites every text node in a paragraph", () => {
		const tree = paragraph("first [1].", " second [2].")
		rehypeCitationMarkers(known)()(tree)
		expect(
			firstChild(tree).children.filter(
				(child) => child.type === "element" && child.tagName === "cite-marker",
			),
		).toHaveLength(2)
	})
})

function source(
	marker: number,
	overrides: Partial<ChatSource> = {},
): ChatSource {
	return {
		marker,
		chunk_id: marker,
		celex: "32023R1805",
		act: "Regulation (EU) 2023/1805",
		citation: "Article 4(1)",
		title: "Greenhouse gas intensity limit",
		text: "The limit applies from 2025.",
		...overrides,
	}
}

describe("citedSources", () => {
	const sources = [source(1), source(7), source(12)]

	it("returns the cited sources with the number the answer shows", () => {
		expect(citedSources("claim [7] and [12].", sources)).toEqual([
			{ source: sources[1], label: 1 },
			{ source: sources[2], label: 2 },
		])
	})

	it("leaves out a retrieved source the answer never cites", () => {
		expect(citedSources("claim [7].", sources)).toEqual([
			{ source: sources[1], label: 1 },
		])
	})

	it("is empty when the answer cites nothing", () => {
		expect(citedSources("claim.", sources)).toEqual([])
	})
})

describe("renumberCitations", () => {
	const sources = [source(1), source(7), source(12)]

	it("rewrites markers as the numbers the reader saw", () => {
		expect(renumberCitations("claim [7] and [12].", sources)).toBe(
			"claim [1] and [2].",
		)
	})

	it("gives a repeated marker the same number each time", () => {
		expect(renumberCitations("claim [7], again [7].", sources)).toBe(
			"claim [1], again [1].",
		)
	})

	it("leaves a marker that was never retrieved alone", () => {
		expect(renumberCitations("claim [9].", sources)).toBe("claim [9].")
	})
})
