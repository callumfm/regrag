import type { paths } from '@/api/schema'

export type HealthResponse =
	paths['/health']['get']['responses']['200']['content']['application/json']

export const API_URL: string =
	import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

export async function fetchHealth(): Promise<HealthResponse> {
	const res = await fetch(`${API_URL}/health`)
	if (!res.ok) throw new Error(`Health request failed: ${res.status}`)
	return res.json()
}
