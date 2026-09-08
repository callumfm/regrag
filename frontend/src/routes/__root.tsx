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

/** Recolours the thinking orb, which draws itself in greyscale: the flood is kept only where
 * the canvas has ink, so the animation's shading survives as its alpha. */
function OrbTint() {
	return (
		<svg aria-hidden className="absolute size-0" role="presentation">
			<title>Orb tint</title>
			<filter id="orb-tint" colorInterpolationFilters="sRGB">
				<feFlood floodColor="var(--primary)" result="tint" />
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
