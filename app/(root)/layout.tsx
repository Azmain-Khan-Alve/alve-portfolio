import { MainNav } from "@/components/common/main-nav";
import { ModeToggle } from "@/components/common/mode-toggle";
import { SiteFooter } from "@/components/common/site-footer";
import { routesConfig } from "@/config/routes";

//=======================(Newly added)
import { Metadata } from "next";
import { pagesConfig } from "@/config/pages";
import { siteConfig } from "@/config/site";

export const metadata: Metadata = {
  title: `${pagesConfig.home.metadata.title} | Modern Next.js Developer Portfolio Template`,
  description: `${pagesConfig.home.metadata.description} This open-source Next.js portfolio template is customizable to showcase your skills and projects.`,
  alternates: {
    canonical: siteConfig.url,
  },
};
//======================

interface MarketingLayoutProps {
  children: React.ReactNode;
}

export default function MarketingLayout({ children }: MarketingLayoutProps) {
  return (
    <div className="flex min-h-screen flex-col">
      <header className="sticky top-0 z-50 w-full bg-background">
        <div className="container flex h-20 items-center justify-between py-6">
          <MainNav items={routesConfig.mainNav} />
          <nav className="flex items-center gap-5">
            {/* <Link
                            href={"https://github.com/namanbarkiya"}
                            target="_blank"
                            className={cn(
                                buttonVariants({
                                    variant: "ghost",
                                    size: "sm",
                                }),
                                "h-8 w-8 px-0"
                            )}
                        >
                            <Icons.gitHub className="w-5 h-5" />
                        </Link> */}
            <ModeToggle />
          </nav>
        </div>
      </header>
      <main className="container flex-1">{children}</main>
      <SiteFooter />
    </div>
  );
}
