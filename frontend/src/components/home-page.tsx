import { useQuery } from '@tanstack/react-query'
import { fetchHealth } from '@/lib/api'

export function HomePage() {
	const health = useQuery({ queryKey: ['health'], queryFn: fetchHealth })

	return (
		<main className="flex min-h-screen flex-col items-center justify-center gap-4">
			<h1 className="font-semibold text-4xl">RegRag</h1>
			<p className="text-neutral-500 text-sm">
				{health.isPending && 'Checking backend…'}
				{health.isError && 'Backend unreachable'}
				{health.data &&
					`Backend ${health.data.status} — v${health.data.version}`}
			</p>
		</main>
	)
}
