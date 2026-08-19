import { Link } from "@tanstack/react-router"
import type { ReactNode } from "react"
import { Button } from "@/components/ui/button"

export function MessagePage({
	title,
	description,
	action,
}: {
	title: string
	description: string
	action?: ReactNode
}) {
	return (
		<div className="flex h-dvh w-full flex-col items-center justify-center gap-y-8 px-12">
			<h1 className="font-semibold text-4xl tracking-tight">{title}</h1>
			<p className="max-w-xl text-center text-muted-foreground">
				{description}
			</p>
			<div className="flex flex-wrap items-center justify-center gap-3">
				{action}
				<Button variant="outline" render={<Link to="/" />}>
					Take me home
				</Button>
			</div>
		</div>
	)
}
