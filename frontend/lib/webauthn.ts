function base64ToArrayBuffer(base64: string) {
  const padding = "=".repeat((4 - (base64.length % 4)) % 4);
  const base64Safe = (base64 + padding).replace(/-/g, "+").replace(/_/g, "/");
  const raw = atob(base64Safe);
  const buffer = new ArrayBuffer(raw.length);
  const bytes = new Uint8Array(buffer);
  for (let i = 0; i < raw.length; i += 1) {
    bytes[i] = raw.charCodeAt(i);
  }
  return buffer;
}

function arrayBufferToBase64(buffer: ArrayBuffer) {
  const bytes = new Uint8Array(buffer);
  let binary = "";
  bytes.forEach((b) => {
    binary += String.fromCharCode(b);
  });
  const base64 = btoa(binary)
    .replace(/\+/g, "-")
    .replace(/\//g, "_")
    .replace(/=+$/, "");
  return base64;
}

export function decodeCreationOptions(options: PublicKeyCredentialCreationOptions) {
  return {
    ...options,
    challenge: base64ToArrayBuffer(options.challenge as unknown as string),
    user: {
      ...options.user,
      id: base64ToArrayBuffer(options.user.id as unknown as string),
    },
    excludeCredentials: options.excludeCredentials?.map((cred) => ({
      ...cred,
      id: base64ToArrayBuffer(cred.id as unknown as string),
    })),
  } as PublicKeyCredentialCreationOptions;
}

export function decodeRequestOptions(options: PublicKeyCredentialRequestOptions) {
  return {
    ...options,
    challenge: base64ToArrayBuffer(options.challenge as unknown as string),
    allowCredentials: options.allowCredentials?.map((cred) => ({
      ...cred,
      id: base64ToArrayBuffer(cred.id as unknown as string),
    })),
  } as PublicKeyCredentialRequestOptions;
}

export function encodeCredential(credential: PublicKeyCredential) {
  const response = credential.response as AuthenticatorResponse;
  const clientExtensionResults = credential.getClientExtensionResults();
  type EncodedResponse = {
    clientDataJSON: string;
    attestationObject?: string;
    authenticatorData?: string;
    signature?: string;
    userHandle?: string | null;
  };

  const credentialData: {
    id: string;
    rawId: string;
    type: string;
    response: EncodedResponse;
    clientExtensionResults: AuthenticationExtensionsClientOutputs;
  } = {
    id: credential.id,
    rawId: arrayBufferToBase64(credential.rawId),
    type: credential.type,
    response: {
      clientDataJSON: arrayBufferToBase64(response.clientDataJSON),
    },
    clientExtensionResults,
  };

  if (response instanceof AuthenticatorAttestationResponse) {
    credentialData.response.attestationObject = arrayBufferToBase64(
      response.attestationObject
    );
  }

  if (response instanceof AuthenticatorAssertionResponse) {
    credentialData.response.authenticatorData = arrayBufferToBase64(
      response.authenticatorData
    );
    credentialData.response.signature = arrayBufferToBase64(response.signature);
    credentialData.response.userHandle = response.userHandle
      ? arrayBufferToBase64(response.userHandle)
      : null;
  }

  return credentialData;
}
