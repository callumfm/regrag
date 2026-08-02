import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen } from '@testing-library/react'
import { HomePage } from './home-page'

test('renders the app name', () => {
	const client = new QueryClient({
		defaultOptions: { queries: { retry: false } },
	})
	render(
		<QueryClientProvider client={client}>
			<HomePage />
		</QueryClientProvider>,
	)
	expect(screen.getByRole('heading', { name: 'RegRag' })).toBeInTheDocument()
})
