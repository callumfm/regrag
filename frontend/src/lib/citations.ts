import type { Element, Root, Text } from "hast"
import { visit } from "unist-util-visit"

const MARKER_PATTERN = /\[(\d+)\]/g
const MARKER_RUN_BEFORE_PUNCTUATION = / ?((?:\[\d+\])+)([.,;:])/g
const FENCE_LINE = /^ {0,3}(`{3,}|~{3,})/
const INLINE_CODE = /(`+)(?:[\s\S]*?\1|[\s\S]*$)/g

/** Superscripts follow punctuation: `claim [1][2].` becomes `claim.[1][2]`. */
export function moveMarkersAfterPunctuation(
	value: string,
	known: ReadonlySet<number>,
): string {
	return value.replace(
		MARKER_RUN_BEFORE_PUNCTUATION,
		(whole: string, run: string, punctuation: string) => {
			const allKnown = [...run.matchAll(MARKER_PATTERN)].every((match) =>
				known.has(Number(match[1])),
			)
			return allKnown ? `${punctuation}${run}` : whole
		},
	)
}

export type CitationSegment =
	| { kind: "text"; value: string }
	| { kind: "marker"; marker: number }

export function splitCitationMarkers(
	raw: string,
	known: ReadonlySet<number>,
): CitationSegment[] {
	const value = moveMarkersAfterPunctuation(raw, known)
	const segments: CitationSegment[] = []
	let cursor = 0
	for (const match of value.matchAll(MARKER_PATTERN)) {
		const marker = Number(match[1])
		if (!known.has(marker)) continue
		if (match.index > cursor) {
			segments.push({ kind: "text", value: value.slice(cursor, match.index) })
		}
		segments.push({ kind: "marker", marker })
		cursor = match.index + match[0].length
	}
	if (cursor < value.length) {
		segments.push({ kind: "text", value: value.slice(cursor) })
	}
	return segments
}

/**
 * Drops fenced blocks and inline spans, so markers inside code are not read as
 * citations. Mirrors the `code` parent that `rehypeCitationMarkers` skips.
 */
function removeMarkdownCode(markdown: string): string {
	const outside: string[] = []
	let fence: string | null = null
	for (const line of markdown.split("\n")) {
		const delimiter = line.match(FENCE_LINE)?.[1]
		if (fence === null) {
			if (delimiter) fence = delimiter
			else outside.push(line)
			continue
		}
		const closes =
			delimiter?.[0] === fence[0] && delimiter.length >= fence.length
		if (closes) fence = null
	}
	return outside.join("\n").replace(INLINE_CODE, "")
}

/** Markers the answer cites, once each, in order of first appearance. */
export function extractCitedMarkers(
	answer: string,
	known: ReadonlySet<number>,
): number[] {
	const cited = new Set<number>()
	for (const match of removeMarkdownCode(answer).matchAll(MARKER_PATTERN)) {
		const marker = Number(match[1])
		if (known.has(marker)) cited.add(marker)
	}
	return [...cited]
}

/**
 * Display number for each cited marker: 1..k by first appearance.
 * Markers are retrieval positions, so an answer citing [7] and [12] reads as 1 and 2.
 */
export function numberCitations(
	answer: string,
	known: ReadonlySet<number>,
): ReadonlyMap<number, number> {
	const cited = extractCitedMarkers(answer, known)
	return new Map(cited.map((marker, index) => [marker, index + 1]))
}

function toHastNode(segment: CitationSegment): Text | Element {
	if (segment.kind === "text") return { type: "text", value: segment.value }
	return {
		type: "element",
		tagName: "cite-marker",
		properties: { marker: String(segment.marker) },
		children: [],
	}
}

export function rehypeCitationMarkers(known: ReadonlySet<number>) {
	return () => (tree: Root) => {
		visit(tree, "text", (node: Text, index, parent) => {
			if (parent === undefined || index === undefined) return
			if (parent.type === "element" && parent.tagName === "code") return
			const segments = splitCitationMarkers(node.value, known)
			if (!segments.some((segment) => segment.kind === "marker")) return
			parent.children.splice(index, 1, ...segments.map(toHastNode))
			return index + segments.length
		})
	}
}
