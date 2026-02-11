export type ApiPagination = {
  count: number;
  next: string | null;
  previous: string | null;
  page: number;
  page_size: number;
  total_pages: number;
};

export type ApiMeta = {
  pagination?: ApiPagination;
} | null;

export type ApiResponse<T> = {
  success: boolean;
  message?: string;
  data: T;
  meta?: ApiMeta;
};

export type UserProfile = {
  id: string;
  email: string;
  first_name?: string | null;
  last_name?: string | null;
  full_name?: string | null;
  phone?: string | null;
  avatar?: string | null;
  date_of_birth?: string | null;
  is_verified?: boolean;
  newsletter_subscribed?: boolean;
  is_staff?: boolean;
  is_superuser?: boolean;
  created_at?: string;
};

export type ProductListItem = {
  id: string;
  name: string;
  slug: string;
  sku?: string | null;
  short_description?: string | null;
  price: string;
  sale_price?: string | null;
  current_price: string;
  currency: string;
  discount_percentage?: string | null;
  is_on_sale: boolean;
  is_in_stock: boolean;
  is_featured?: boolean;
  is_bestseller?: boolean;
  is_new_arrival?: boolean;
  average_rating?: number | null;
  reviews_count?: number | null;
  views_count?: number | null;
  primary_image?: string | null;
  primary_category_name?: string | null;
};

export type ProductImage = {
  id: string;
  image: string;
  alt_text?: string | null;
  is_primary?: boolean;
  ordering?: number;
};

export type ProductVariant = {
  id: string;
  sku?: string | null;
  price?: string | null;
  stock_quantity?: number | null;
  is_default?: boolean;
  current_price?: string | null;
  option_values?: Array<{
    id: string;
    option: { id: string; name: string; slug: string };
    value: string;
  }>;
};

export type ProductDetail = ProductListItem & {
  description?: string | null;
  images?: ProductImage[];
  variants?: ProductVariant[];
  categories?: Array<{ id: string; name: string; slug: string }>;
  primary_category?: { id: string; name: string; slug: string } | null;
  tags?: Array<{ id: string; name: string }>;
  attributes?: Array<{ id: string; attribute: { id: string; name: string; slug: string }; value: string }>;
  is_low_stock?: boolean;
  available_stock?: number | null;
  badges?: ProductBadge[];
  material_breakdown?: Record<string, number | string> | null;
  meta_title?: string | null;
  meta_description?: string | null;
  schema_org?: Record<string, unknown> | null;
  breadcrumbs?: Array<{ id: string; name: string; slug: string }> | null;
  weight?: string | number | null;
  length?: string | number | null;
  width?: string | number | null;
  height?: string | number | null;
  shipping_material?: { id: string; name?: string | null; eco_score?: number | null; notes?: string | null; packaging_weight?: number | null } | null;
  eco_certifications?: Array<{ id: string; name: string; slug?: string | null; issuer?: string | null }> | null;
  carbon_footprint_kg?: number | null;
  recycled_content_percentage?: number | null;
  sustainability_score?: number | null;
  assets_3d?: Array<{ id: string; file?: string | null; poster_image?: string | null; poster_alt?: string | null; is_primary?: boolean | null; is_ar_compatible?: boolean | null; ar_quicklook_url?: string | null }> | null;
  is_ar_compatible?: boolean | null;
  is_mobile_optimized?: boolean | null;
};

export type ProductBadge = {
  id: string;
  name: string;
  slug: string;
  css_class?: string | null;
  start?: string | null;
  end?: string | null;
  priority?: number | null;
};

export type ReviewImage = {
  id: string;
  image: string;
};

export type Review = {
  id: string;
  user_name?: string | null;
  rating: number;
  title?: string | null;
  body?: string | null;
  verified_purchase?: boolean;
  helpful_votes?: number;
  moderation_status?: string;
  images?: ReviewImage[];
  created_at?: string | null;
  updated_at?: string | null;
};

export type ReviewSummary = {
  total: number;
  average: number;
  distribution: Record<string, number>;
};

export type ProductReviewsResponse = {
  reviews: Review[];
  summary: ReviewSummary;
  total: number;
  page: number;
  total_pages: number;
};

export type ProductQuestion = {
  id: string;
  product: string;
  user?: string | null;
  user_name?: string | null;
  question_text: string;
  status?: string | null;
  created_at?: string | null;
  answers?: ProductAnswer[];
};

export type ProductAnswer = {
  id: string;
  question: string;
  user?: string | null;
  user_name?: string | null;
  answer_text: string;
  status?: string | null;
  created_at?: string | null;
};

export type CustomerPhoto = {
  id: string;
  product: string;
  product_name?: string | null;
  user?: string | null;
  user_name?: string | null;
  image: string;
  description?: string | null;
  status?: string | null;
  created_at?: string | null;
};

export type ProductFilterResponse = {
  price_range: {
    min: number;
    max: number;
    currency: string;
    currency_symbol: string;
  };
  attributes: Record<string, { slug: string; values: string[] }>;
  tags: Array<{ name: string; slug: string }>;
  has_on_sale?: boolean;
  has_free_shipping?: boolean;
};

export type CartItem = {
  id: string;
  product_id: string;
  product_name: string;
  product_slug: string;
  product_image?: string | null;
  variant_id?: string | null;
  variant_name?: string | null;
  quantity: number;
  unit_price: string;
  total: string;
  price_at_add?: string | null;
  in_stock: boolean;
  gift_wrap?: boolean;
  gift_message?: string | null;
  created_at?: string | null;
};

export type Cart = {
  id: string;
  items: CartItem[];
  item_count: number;
  subtotal: string;
  discount_amount: string;
  total: string;
  coupon_code?: string | null;
  currency: string;
  updated_at?: string;
};

export type CartSummary = {
  id?: string;
  items?: CartItem[];
  item_count?: number;
  subtotal: string;
  discount_amount: string;
  shipping_cost?: string;
  tax_amount?: string;
  gift_wrap_cost?: string;
  gift_wrap_amount?: string;
  total: string;
  coupon_code?: string | null;
  formatted_subtotal?: string;
  formatted_discount?: string;
  formatted_shipping?: string;
  formatted_tax?: string;
  formatted_gift_wrap?: string;
  formatted_gift_wrap_amount?: string;
  formatted_total?: string;
  payment_fee_amount?: string;
  payment_fee_label?: string | null;
  formatted_payment_fee?: string;
  currency?: string | null;
  currency_code?: string | null;
  currency_symbol?: string | null;
  shipping_estimate?: boolean;
  shipping_selected?: boolean;
  shipping_zone?: string | null;
  shipping_method_name?: string | null;
  shipping_method_code?: string | null;
  shipping_rate_id?: string | null;
  shipping_estimate_label?: string | null;
  pickup_location_id?: string | null;
  pickup_location_name?: string | null;
  gift_wrap_label?: string | null;
  gift_wrap_enabled?: boolean;
  tax_rate?: string | null;
};

export type WishlistItem = {
  id: string;
  product_id: string;
  product_name: string;
  product_slug: string;
  product_image?: string | null;
  current_price?: string | null;
  price_at_add?: string | null;
  in_stock?: boolean;
  added_at?: string;
};

export type OrderListItem = {
  id: string;
  order_number: string;
  status: string;
  status_display?: string;
  total: string;
  item_count: number;
  created_at: string;
};

export type PageDetail = {
  id: string;
  title: string;
  slug: string;
  content?: string | null;
  excerpt?: string | null;
  featured_image?: string | null;
  meta_title?: string | null;
  meta_description?: string | null;
  created_at?: string;
  updated_at?: string;
};

export type MenuPage = {
  id: string;
  title: string;
  slug: string;
  url?: string | null;
  menu_order?: number | null;
};

export type SiteSettings = {
  site_name?: string | null;
  tagline?: string | null;
  site_tagline?: string | null;
  site_description?: string | null;
  footer_text?: string | null;
  copyright_text?: string | null;
  contact_email?: string | null;
  support_email?: string | null;
  contact_phone?: string | null;
  contact_address?: string | null;
  address?: string | null;
  facebook_url?: string | null;
  instagram_url?: string | null;
  twitter_url?: string | null;
  linkedin_url?: string | null;
  youtube_url?: string | null;
  tiktok_url?: string | null;
};

export type ContactSettings = {
  general_email?: string | null;
  support_email?: string | null;
  sales_email?: string | null;
  phone?: string | null;
  business_hours_note?: string | null;
  social_links?: Record<string, string | null> | null;
};

export type SocialLink = {
  label: string;
  url: string;
};

export type Category = {
  id: string;
  name: string;
  slug: string;
};

export type FAQItem = {
  id: string;
  question: string;
  answer: string;
  category?: string | null;
};

export type Address = {
  id: string;
  address_type?: string | null;
  full_name?: string | null;
  phone?: string | null;
  address_line_1?: string | null;
  address_line_2?: string | null;
  city?: string | null;
  state?: string | null;
  postal_code?: string | null;
  country?: string | null;
  is_default?: boolean;
  full_address?: string | null;
};

export type Country = {
  id: string;
  code: string;
  name: string;
  flag_emoji?: string | null;
  phone_code?: string | null;
};

export type OrderItem = {
  id: string;
  product_name: string;
  product_sku?: string | null;
  variant_name?: string | null;
  product_image?: string | null;
  unit_price: string;
  quantity: number;
  line_total?: string;
};

export type OrderStatusHistory = {
  id: string;
  old_status?: string | null;
  new_status?: string | null;
  notes?: string | null;
  created_at?: string | null;
};

export type OrderDetail = {
  id: string;
  order_number: string;
  status: string;
  status_display?: string;
  email?: string | null;
  phone?: string | null;
  subtotal?: string | null;
  discount?: string | null;
  tax?: string | null;
  total: string;
  shipping_cost?: string | null;
  currency?: string | null;
  exchange_rate?: string | null;
  payment_method?: string | null;
  payment_status?: string | null;
  payment_fee_amount?: string | null;
  payment_fee_label?: string | null;
  gift_wrap?: boolean | null;
  gift_wrap_cost?: string | null;
  coupon_code?: string | null;
  customer_notes?: string | null;
  item_count?: number;
  items?: OrderItem[];
  status_history?: OrderStatusHistory[];
  created_at?: string | null;
  shipping_address?: Record<string, string | null>;
  billing_address?: Record<string, string | null>;
  shipping_method?: string | null;
  shipping_method_display?: string | null;
  payment_method_display?: string | null;
  pickup_location?: {
    id: string;
    name: string;
    full_address?: string | null;
    phone?: string | null;
    email?: string | null;
  } | null;
  tracking_number?: string | null;
  tracking_url?: string | null;
};

export type NotificationItem = {
  id: string;
  type?: string | null;
  category?: string | null;
  priority?: string | null;
  status?: string | null;
  delivery_status?: string | null;
  title?: string | null;
  message?: string | null;
  url?: string | null;
  is_read?: boolean;
  read_at?: string | null;
  created_at?: string | null;
  channels_requested?: string[] | null;
  channels_sent?: string[] | null;
  metadata?: Record<string, unknown> | null;
};

export type NotificationPreference = {
  email_enabled?: boolean;
  email_order_updates?: boolean;
  email_shipping_updates?: boolean;
  email_promotions?: boolean;
  email_newsletter?: boolean;
  email_reviews?: boolean;
  email_price_drops?: boolean;
  email_back_in_stock?: boolean;
  sms_enabled?: boolean;
  sms_order_updates?: boolean;
  sms_shipping_updates?: boolean;
  sms_promotions?: boolean;
  push_enabled?: boolean;
  push_order_updates?: boolean;
  push_promotions?: boolean;
  digest_frequency?: string;
  quiet_hours_start?: string | null;
  quiet_hours_end?: string | null;
  timezone?: string | null;
  marketing_opt_in?: boolean;
  transactional_opt_in?: boolean;
  per_type_overrides?: Record<string, unknown> | null;
};

export type UserPreferences = {
  language?: string | null;
  currency?: string | null;
  timezone?: string | null;
  theme?: string | null;
  email_notifications?: boolean;
  sms_notifications?: boolean;
  push_notifications?: boolean;
  notify_order_updates?: boolean;
  notify_promotions?: boolean;
  notify_price_drops?: boolean;
  notify_back_in_stock?: boolean;
  notify_recommendations?: boolean;
  allow_tracking?: boolean;
  share_data_for_ads?: boolean;
  reduce_motion?: boolean;
  high_contrast?: boolean;
  large_text?: boolean;
};

export type UserSession = {
  id: string;
  session_type?: string | null;
  ip_address?: string | null;
  device_type?: string | null;
  device_brand?: string | null;
  device_model?: string | null;
  browser?: string | null;
  browser_version?: string | null;
  os?: string | null;
  os_version?: string | null;
  country?: string | null;
  country_code?: string | null;
  region?: string | null;
  city?: string | null;
  started_at?: string | null;
  last_activity?: string | null;
  revoked_at?: string | null;
  is_active?: boolean;
  is_current?: boolean;
};

export type MfaStatus = {
  enabled?: boolean;
  methods?: string[];
  backup_codes_remaining?: number;
  passkey_count?: number;
};

export type WebAuthnCredential = {
  id: string;
  nickname?: string | null;
  transports?: string[];
  last_used_at?: string | null;
  created_at?: string | null;
  is_active?: boolean;
};

export type DataExportJob = {
  id: string;
  status?: string | null;
  requested_at?: string | null;
  completed_at?: string | null;
  expires_at?: string | null;
  file?: string | null;
  error_message?: string | null;
};

export type AccountDeletionStatus = {
  status?: string | null;
  requested_at?: string | null;
  scheduled_for?: string | null;
  processed_at?: string | null;
  cancelled_at?: string | null;
  reason?: string | null;
};

export type SubscriptionPlan = {
  id: string;
  name: string;
  description?: string | null;
  interval?: string | null;
  price_amount?: string | null;
  currency?: string | null;
  trial_period_days?: number | null;
  active?: boolean;
  metadata?: Record<string, unknown> | null;
};

export type Subscription = {
  id: string;
  plan?: SubscriptionPlan;
  plan_id?: string;
  status?: string | null;
  quantity?: number | null;
  current_period_start?: string | null;
  current_period_end?: string | null;
  next_billing_at?: string | null;
  canceled_at?: string | null;
  metadata?: Record<string, unknown> | null;
  recurring_charges?: Array<Record<string, unknown>>;
};

export type PreorderCategory = {
  id: string;
  name: string;
  slug: string;
  description?: string | null;
  icon?: string | null;
  image?: string | null;
  base_price?: string | number | null;
  deposit_percentage?: number | null;
  min_production_days?: number | null;
  max_production_days?: number | null;
  requires_design?: boolean | null;
  requires_approval?: boolean | null;
  allow_rush_order?: boolean | null;
  rush_order_fee_percentage?: number | null;
  min_quantity?: number | null;
  max_quantity?: number | null;
  is_active?: boolean | null;
  preorder_count?: number | null;
  options?: PreorderOption[];
};

export type PreorderOptionChoice = {
  id: string;
  value: string;
  display_name: string;
  price_modifier?: string | number | null;
  color_code?: string | null;
  image?: string | null;
  order?: number | null;
  is_active?: boolean | null;
};

export type PreorderOption = {
  id: string;
  name: string;
  description?: string | null;
  option_type: string;
  is_required?: boolean | null;
  min_length?: number | null;
  max_length?: number | null;
  price_modifier?: string | number | null;
  placeholder?: string | null;
  help_text?: string | null;
  order?: number | null;
  is_active?: boolean | null;
  choices?: PreorderOptionChoice[];
};

export type PreorderTemplate = {
  id: string;
  name: string;
  slug: string;
  description?: string | null;
  category?: string | null;
  category_name?: string | null;
  image?: string | null;
  default_quantity?: number | null;
  base_price?: string | number | null;
  estimated_days?: number | null;
  default_options?: Record<string, unknown> | null;
  is_active?: boolean | null;
  is_featured?: boolean | null;
  order?: number | null;
  use_count?: number | null;
};

export type PreorderOptionValue = {
  id: string;
  option: string;
  option_name?: string | null;
  option_type?: string | null;
  text_value?: string | null;
  number_value?: string | null;
  choice_value?: string | null;
  boolean_value?: boolean | null;
  date_value?: string | null;
  file_value?: string | null;
  price_modifier_applied?: string | null;
  display_value?: string | null;
};

export type PreorderDesign = {
  id: string;
  file?: string | null;
  original_filename?: string | null;
  design_type?: string | null;
  is_approved?: boolean | null;
  approved_at?: string | null;
  notes?: string | null;
  created_at?: string | null;
};

export type PreorderReference = {
  id: string;
  file?: string | null;
  original_filename?: string | null;
  description?: string | null;
  created_at?: string | null;
};

export type PreorderMessage = {
  id: string;
  sender_name?: string | null;
  subject?: string | null;
  message?: string | null;
  is_from_customer?: boolean | null;
  is_from_system?: boolean | null;
  attachment?: string | null;
  is_read?: boolean | null;
  created_at?: string | null;
};

export type PreorderRevision = {
  id: string;
  revision_number?: number | null;
  description?: string | null;
  status?: string | null;
  additional_cost?: string | null;
  created_at?: string | null;
  completed_at?: string | null;
};

export type PreorderQuote = {
  id: string;
  quote_number?: string | null;
  base_price?: string | null;
  customization_cost?: string | null;
  rush_fee?: string | null;
  discount?: string | null;
  shipping?: string | null;
  tax?: string | null;
  total?: string | null;
  valid_from?: string | null;
  valid_until?: string | null;
  is_expired?: boolean | null;
  status?: string | null;
  terms?: string | null;
  notes?: string | null;
  estimated_production_days?: number | null;
  estimated_delivery_date?: string | null;
  created_at?: string | null;
  sent_at?: string | null;
  responded_at?: string | null;
};

export type PreorderPayment = {
  id: string;
  payment_type?: string | null;
  amount?: string | null;
  currency?: string | null;
  status?: string | null;
  payment_method?: string | null;
  transaction_id?: string | null;
  notes?: string | null;
  receipt_url?: string | null;
  created_at?: string | null;
  paid_at?: string | null;
};

export type PreorderStatusHistory = {
  id: string;
  from_status?: string | null;
  from_status_display?: string | null;
  to_status?: string | null;
  to_status_display?: string | null;
  changed_by_name?: string | null;
  notes?: string | null;
  created_at?: string | null;
  is_system?: boolean | null;
};

export type Preorder = {
  id: string;
  preorder_number?: string | null;
  status?: string | null;
  status_display?: string | null;
  priority?: string | null;
  priority_display?: string | null;
  title?: string | null;
  description?: string | null;
  quantity?: number | null;
  estimated_price?: string | null;
  final_price?: string | null;
  total_amount?: string | null;
  discount_amount?: string | null;
  tax_amount?: string | null;
  shipping_cost?: string | null;
  deposit_required?: string | null;
  deposit_paid?: string | null;
  amount_paid?: string | null;
  amount_remaining?: string | null;
  is_fully_paid?: boolean | null;
  deposit_is_paid?: boolean | null;
  currency?: string | null;
  is_rush_order?: boolean | null;
  rush_order_fee?: string | null;
  requested_delivery_date?: string | null;
  estimated_completion_date?: string | null;
  created_at?: string | null;
  submitted_at?: string | null;
  quoted_at?: string | null;
  approved_at?: string | null;
  production_started_at?: string | null;
  completed_at?: string | null;
  shipped_at?: string | null;
  delivered_at?: string | null;
  category?: PreorderCategory | string | null;
  category_name?: string | null;
  full_name?: string | null;
  email?: string | null;
  phone?: string | null;
  special_instructions?: string | null;
  customer_notes?: string | null;
  is_gift?: boolean | null;
  gift_wrap?: boolean | null;
  gift_message?: string | null;
  quote_valid_until?: string | null;
  quote_notes?: string | null;
  items?: Array<Record<string, unknown>>;
  option_values?: PreorderOptionValue[];
  designs?: PreorderDesign[];
  references?: PreorderReference[];
  payments?: PreorderPayment[];
  messages?: PreorderMessage[];
  revisions?: PreorderRevision[];
  quotes?: PreorderQuote[];
  status_history?: PreorderStatusHistory[];
  shipping_first_name?: string | null;
  shipping_last_name?: string | null;
  shipping_address_line_1?: string | null;
  shipping_address_line_2?: string | null;
  shipping_city?: string | null;
  shipping_state?: string | null;
  shipping_postal_code?: string | null;
  shipping_country?: string | null;
  shipping_method?: string | null;
  tracking_number?: string | null;
  tracking_url?: string | null;
};

export type PreorderPriceEstimate = {
  base_price: string;
  options_price: string;
  rush_fee: string;
  subtotal: string;
  total: string;
  deposit_required: string;
  deposit_percentage?: number | null;
  currency: string;
};

export type PreorderStatistics = {
  total?: number;
  draft?: number;
  pending?: number;
  in_production?: number;
  completed?: number;
  delivered?: number;
  total_value?: string | number | null;
  total_paid?: string | number | null;
};

export type Collection = {
  id: string;
  name: string;
  slug: string;
  description?: string | null;
  image?: string | null;
};

export type Bundle = {
  id: string;
  name: string;
  slug: string;
  description?: string | null;
  image?: string | null;
};

export type Artisan = {
  id: string;
  name: string;
  slug: string;
  bio?: string | null;
  avatar?: string | null;
};

export type PaymentMethod = {
  id: string;
  name?: string | null;
  description?: string | null;
  active?: boolean;
};

export type ShippingMethod = {
  id: string;
  name?: string | null;
  price?: string | null;
  estimated_days?: number | null;
};

export type CheckoutSession = {
  id: string;
  current_step?: string | null;
  email?: string | null;
  shipping_first_name?: string | null;
  shipping_last_name?: string | null;
  shipping_company?: string | null;
  shipping_email?: string | null;
  shipping_phone?: string | null;
  shipping_address_line_1?: string | null;
  shipping_address_line_2?: string | null;
  shipping_city?: string | null;
  shipping_state?: string | null;
  shipping_postal_code?: string | null;
  shipping_country?: string | null;
  billing_first_name?: string | null;
  billing_last_name?: string | null;
  billing_company?: string | null;
  billing_address_line_1?: string | null;
  billing_address_line_2?: string | null;
  billing_city?: string | null;
  billing_state?: string | null;
  billing_postal_code?: string | null;
  billing_country?: string | null;
  billing_same_as_shipping?: boolean;
  shipping_method?: string | null;
  shipping_cost?: string | null;
  payment_method?: string | null;
  payment_fee_amount?: string | null;
  payment_fee_label?: string | null;
  subtotal?: string | null;
  discount_amount?: string | null;
  tax_amount?: string | null;
  total?: string | null;
  coupon_code?: string | null;
  order_notes?: string | null;
  delivery_instructions?: string | null;
  is_gift?: boolean;
  gift_message?: string | null;
  gift_wrap?: boolean;
  gift_wrap_cost?: string | null;
  pickup_location?: {
    id: string;
    name: string;
    full_address?: string | null;
    phone?: string | null;
    email?: string | null;
    pickup_fee?: string | null;
    min_pickup_time_hours?: number | null;
    max_hold_days?: number | null;
  } | null;
  cart_summary?: {
    item_count?: number;
    subtotal?: string;
    total?: string;
  } | null;
  created_at?: string | null;
  expires_at?: string | null;
};

export type ShippingMethodOption = {
  id?: string;
  rate_id?: string;
  method_id?: string;
  code?: string;
  name: string;
  description?: string | null;
  carrier?: {
    id?: string | null;
    name?: string | null;
    logo?: string | null;
  } | null;
  rate?: number;
  rate_display?: string;
  currency?: {
    code?: string;
    symbol?: string;
    decimal_places?: number;
  } | null;
  is_free?: boolean;
  delivery_estimate?: string | null;
  min_days?: number | null;
  max_days?: number | null;
  is_express?: boolean;
  requires_signature?: boolean;
  zone?: { id?: string; name?: string } | null;
};

export type PaymentGateway = {
  code: string;
  name: string;
  description?: string | null;
  icon_url?: string | null;
  icon_class?: string | null;
  color?: string | null;
  fee_type?: string | null;
  fee_amount?: number | null;
  fee_amount_converted?: number | null;
  fee_text?: string | null;
  instructions?: string | null;
  public_key?: string | null;
  requires_client?: boolean;
};

export type SavedPaymentMethod = {
  id: string;
  type?: string | null;
  type_display?: string | null;
  display_name?: string | null;
  card_brand?: string | null;
  card_last_four?: string | null;
  card_exp_month?: number | null;
  card_exp_year?: number | null;
  paypal_email?: string | null;
  is_default?: boolean;
  created_at?: string | null;
};

export type CheckoutValidationIssue = {
  type?: string;
  message?: string;
  item_id?: string;
  product_name?: string;
  available?: number;
  old_price?: string;
  new_price?: string;
  minimum?: string;
  current?: string;
  max_quantity?: number;
};

export type CheckoutValidation = {
  is_valid: boolean;
  issues: CheckoutValidationIssue[];
  warnings: CheckoutValidationIssue[];
  valid_items?: string[];
  issue_count?: number;
  warning_count?: number;
};

export type ShippingRateResponse = {
  methods: ShippingMethodOption[];
  zone?: { id: string; name: string } | null;
};

export type GiftOptionsResponse = {
  success?: boolean;
  message?: string;
  gift_state?: {
    is_gift?: boolean;
    gift_message?: string;
    gift_wrap?: boolean;
    gift_wrap_cost?: string;
  };
  gift_wrap_amount?: string;
  formatted_gift_wrap_amount?: string;
  formatted_gift_wrap_cost?: string;
  gift_wrap_label?: string;
  gift_wrap_enabled?: boolean;
};

export type StoreLocation = {
  id: string;
  name: string;
  slug?: string | null;
  address?: string | null;
  address_line1?: string | null;
  address_line2?: string | null;
  full_address?: string | null;
  city?: string | null;
  state?: string | null;
  postal_code?: string | null;
  country?: string | null;
  phone?: string | null;
  email?: string | null;
  hours?: Record<string, string | null> | string | null;
  operating_hours?: string | null;
  pickup_fee?: string | number | null;
  min_pickup_time_hours?: number | null;
  max_hold_days?: number | null;
  latitude?: string | number | null;
  longitude?: string | number | null;
};
