import dynamic from "next/dynamic";

const CartPageView = dynamic(
  () => import("@/components/cart/CartPage").then((mod) => mod.CartPage)
);

export default function CartPage() {
  return <CartPageView />;
}
