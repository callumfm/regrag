import { expect, test } from 'vitest'
import { extractCitedMarkers, splitCitationMarkers } from './markers'

const KNOWN = new Set([1, 2])

test('splits a known marker out of the surrounding text', () => {
	expect(splitCitationMarkers('Ships must comply [1].', KNOWN)).toEqual([
		{ kind: 'text', value: 'Ships must comply ' },
		{ kind: 'marker', marker: 1 },
		{ kind: 'text', value: '.' },
	])
})

test('leaves a marker the sources never delivered as literal text', () => {
	expect(splitCitationMarkers('Invented [7].', KNOWN)).toEqual([
		{ kind: 'text', value: 'Invented [7].' },
	])
})

test('leaves a half-streamed marker untouched', () => {
	expect(splitCitationMarkers('Ships must comply [1', KNOWN)).toEqual([
		{ kind: 'text', value: 'Ships must comply [1' },
	])
})

test('splits every occurrence of a repeated marker', () => {
	expect(splitCitationMarkers('[1] and again [1]', KNOWN)).toEqual([
		{ kind: 'marker', marker: 1 },
		{ kind: 'text', value: ' and again ' },
		{ kind: 'marker', marker: 1 },
	])
})

test('lists cited markers once each, in order', () => {
	expect(extractCitedMarkers('[2] then [1] then [2]', KNOWN)).toEqual([1, 2])
})

test('does not count markers the sources never delivered', () => {
	expect(extractCitedMarkers('[1] and [7]', KNOWN)).toEqual([1])
})
