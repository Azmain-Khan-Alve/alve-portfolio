import { Icons } from "@/components/common/icons";

interface SocialInterface {
  name: string;
  username: string;
  icon: any;
  link: string;
}

export const SocialLinks: SocialInterface[] = [
  {
    name: "Github",
    username: "Azmain Khan Alve",
    icon: Icons.gitHub,
    link: "https://github.com/azmain-khan-alve",
  },
  {
    name: "LinkedIn",
    username: "MD. AZMAIN KHAN ALVE",
    icon: Icons.linkedin,
    link: "https://www.linkedin.com/in/md-azmain-khan-alve/",
  },
  {
    name: "Twitter",
    username: "@Alve699",
    icon: Icons.twitter,
    link: "https://x.com/Alve699",
  },
  {
    name: "Gmail",
    username: "azmain.alve",
    icon: Icons.gmail,
    link: "mailto:azmain.khan.alve.699@gmail.com",
  },
];
