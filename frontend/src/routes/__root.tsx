import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { createRootRoute, Outlet } from "@tanstack/react-router"
import { GlobalError } from "@/components/shared/errors/global-error"
import { NotFound } from "@/components/shared/errors/not-found"

export const queryClient = new QueryClient()

export const Route = createRootRoute({
	component: RootLayout,
	notFoundComponent: NotFound,
	errorComponent: GlobalError,
})

/** Paints the thinking orb in the text colour. It draws itself in greyscale, so flooding it
 * and keeping the flood only where the canvas has ink leaves the shading in the alpha. */
function OrbTint() {
	return (
		<svg aria-hidden className="absolute size-0" role="presentation">
			<title>Orb tint</title>
			<filter id="orb-tint" colorInterpolationFilters="sRGB">
				<feFlood floodColor="var(--foreground)" result="tint" />
				<feComposite in="tint" in2="SourceGraphic" operator="in" />
			</filter>
		</svg>
	)
}

function RootLayout() {
	return (
		<QueryClientProvider client={queryClient}>
			<OrbTint />
			<Outlet />
		</QueryClientProvider>
	)
}
