import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import {
	createMemoryHistory,
	createRootRoute,
	createRouter,
	RouterProvider,
} from '@tanstack/react-router'
import { render, screen } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'
import type { HealthResponse } from '@/lib/api'
import { HomePage } from './home-page'

async function renderPage() {
	const client = new QueryClient({
		defaultOptions: { queries: { retry: false } },
	})
	const router = createRouter({
		routeTree: createRootRoute({ component: HomePage }),
		history: createMemoryHistory({ initialEntries: ['/'] }),
	})
	await router.load()
	render(
		<QueryClientProvider client={client}>
			<RouterProvider router={router} />
		</QueryClientProvider>,
	)
}

function stubHealth(body: HealthResponse) {
	vi.stubGlobal(
		'fetch',
		vi.fn().mockResolvedValue({ ok: true, json: async () => body }),
	)
}

afterEach(() => {
	vi.unstubAllGlobals()
})

test('renders the app name', async () => {
	await renderPage()
	expect(screen.getByRole('heading', { name: 'RegRag' })).toBeInTheDocument()
})

test('reports an ok backend with its version', async () => {
	stubHealth({ version: '0.1.0', database: 'ok', status: 'ok' })
	await renderPage()
	expect(await screen.findByText('Backend ok — v0.1.0')).toBeInTheDocument()
})

test('reports a degraded backend instead of claiming it is ok', async () => {
	stubHealth({ version: '0.1.0', database: 'error', status: 'degraded' })
	await renderPage()
	expect(
		await screen.findByText('Backend degraded — v0.1.0'),
	).toBeInTheDocument()
})
