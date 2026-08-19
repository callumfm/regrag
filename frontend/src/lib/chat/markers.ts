import type { Element, Root, Text } from 'hast'
import { visit } from 'unist-util-visit'

const MARKER_PATTERN = /\[(\d+)\]/g
const MARKER_RUN_BEFORE_PUNCTUATION = / ?((?:\[\d+\])+)([.,;:])/g

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
	| { kind: 'text'; value: string }
	| { kind: 'marker'; marker: number }

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
			segments.push({ kind: 'text', value: value.slice(cursor, match.index) })
		}
		segments.push({ kind: 'marker', marker })
		cursor = match.index + match[0].length
	}
	if (cursor < value.length) {
		segments.push({ kind: 'text', value: value.slice(cursor) })
	}
	return segments
}

/** Markers the answer cites, once each, in order of first appearance. */
export function extractCitedMarkers(
	answer: string,
	known: ReadonlySet<number>,
): number[] {
	const cited = new Set<number>()
	for (const match of answer.matchAll(MARKER_PATTERN)) {
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
	if (segment.kind === 'text') return { type: 'text', value: segment.value }
	return {
		type: 'element',
		tagName: 'cite-marker',
		properties: { marker: String(segment.marker) },
		children: [],
	}
}

export function rehypeCitationMarkers(known: ReadonlySet<number>) {
	return () => (tree: Root) => {
		visit(tree, 'text', (node: Text, index, parent) => {
			if (parent === undefined || index === undefined) return
			if (parent.type === 'element' && parent.tagName === 'code') return
			const segments = splitCitationMarkers(node.value, known)
			if (!segments.some((segment) => segment.kind === 'marker')) return
			parent.children.splice(index, 1, ...segments.map(toHastNode))
			return index + segments.length
		})
	}
}
