# Cloudflare Tunnel Relay (Remote Access via kerfbit.dev)

Exposes `metrics_api_server`, `mns_server`, `registry_server` (HTTP API only), and
`chatbot_api_server` to the Android tablet apps when off the home LAN, by relaying
through `kerfbit.dev` via [Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/)
and gating access with [Cloudflare Access](https://developers.cloudflare.com/cloudflare-one/policies/access/)
service tokens.

None of the four backend services have authentication or TLS of their own — that is
fine on a trusted LAN, but not once traffic can originate from the public internet.
This setup does not change any backend code; Access rejects unauthenticated requests
at Cloudflare's edge before they ever reach a tunnel connector or the LAN.

**Out of scope:** the dataset registry's embedded FTP transfer (control port 2121,
PASV range 50000-50099) is not tunneled. It's used for dataset transport between
trainer machines, not by the tablet apps, and doesn't fit cleanly through a single
HTTP ingress rule.

## Topology

|Machine|Tunnel|Public hostname|Local target|
|---|---|---|---|
|192.168.1.16|`adai-storage-tunnel`|`metrics.kerfbit.dev`|`http://localhost:8081`|
|192.168.1.16|`adai-storage-tunnel`|`mns.kerfbit.dev`|`http://localhost:8083`|
|192.168.1.16|`adai-storage-tunnel`|`registry.kerfbit.dev`|`http://localhost:8082`|
|192.168.1.24|`adai-chat-tunnel`|`chat.kerfbit.dev`|`http://localhost:8080`|

Each machine runs its own `cloudflared` connector making an outbound-only connection
to Cloudflare's edge — no router port-forwarding, no public IP exposure on either
machine.

## Prerequisites

- `kerfbit.dev`'s DNS zone must already be on Cloudflare (required for both Tunnel

  routing and Access). Confirm in the Cloudflare dashboard before proceeding.

- A Cloudflare Zero Trust plan that includes Access (the free tier covers this scale).

## 1. Install cloudflared

On both 192.168.1.16 and 192.168.1.24, install the `cloudflared` binary per
[Cloudflare's install instructions](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/)
for the host OS/architecture. Do not use `install_cloudflared.sh` (below) until the
binary is present — the script installs the systemd unit and config, not the binary
itself.

## 2. Authenticate and create the tunnels

```bash
cloudflared tunnel login

# On 192.168.1.16:
cloudflared tunnel create adai-storage-tunnel

# On 192.168.1.24:
cloudflared tunnel create adai-chat-tunnel
```

Each `create` prints a tunnel UUID and writes a credentials JSON file to
`~/.cloudflared/<UUID>.json`. **This file is the tunnel's private key material —
never commit it to git, never copy it outside the two machines that need it.**

## 3. Write ingress config

Copy the appropriate template from `scripts/cloudflared/` on each machine, fill in
the real tunnel UUID and credentials-file path, and place it at
`/etc/cloudflared/config.yml` (or wherever `install_cloudflared.sh` is told to
install it — see step 5).

- 192.168.1.16: `scripts/cloudflared/config-storage.yml.template`
- 192.168.1.24: `scripts/cloudflared/config-chat.yml.template`

## 4. Create DNS records

```bash
# On 192.168.1.16:
cloudflared tunnel route dns adai-storage-tunnel metrics.kerfbit.dev
cloudflared tunnel route dns adai-storage-tunnel mns.kerfbit.dev
cloudflared tunnel route dns adai-storage-tunnel registry.kerfbit.dev

# On 192.168.1.24:
cloudflared tunnel route dns adai-chat-tunnel chat.kerfbit.dev
```

Each command creates a proxied CNAME automatically — no manual DNS editing needed.

## 5. Install cloudflared as a systemd service

```bash
sudo ./scripts/cloudflared/install_cloudflared.sh \
  --tunnel-name adai-storage-tunnel \
  --config-src /path/to/filled-in/config-storage.yml \
  --yes
```

Run the equivalent on 192.168.1.24 with `--tunnel-name adai-chat-tunnel` and the
chat config. See `install_cloudflared.sh --help` for all flags. The script creates a
dedicated `cloudflared` system user (not `adai`) — see the script's header comment
for why.

## 6. Configure Cloudflare Access

In the Cloudflare Zero Trust dashboard:

1. **Access → Applications → Add an application → Self-hosted.** Add all four

   hostnames (`metrics.kerfbit.dev`, `mns.kerfbit.dev`, `registry.kerfbit.dev`,
   `chat.kerfbit.dev`) as public hostnames under one application. (Splitting into
   four separate applications later, for per-service revocation, is a
   straightforward refinement — not required to start.)

2. **Add a policy** with action **Service Auth** (not an interactive "Allow" rule

   for humans) requiring a valid service token.

3. **Access → Service Auth → Service Tokens → Create.** Name it

   `adai-tablet-relay`. The Client ID and Client Secret are shown once — save them
   in a password manager. **Never commit them to git.** One shared token for both
   Android apps is fine to start; separate tokens per app is an easy upgrade later
   if you want independent revocation.

4. Attach the new service token to the policy from step 2.

## Verification

From an external network (e.g. a phone hotspot — **not** the home Wi-Fi, to prove
the public path actually works):

```bash
# Expect a 401/403 / Access login redirect — no token headers sent
curl -i https://metrics.kerfbit.dev/health

# Expect the real metrics_api_server health JSON to pass through
curl -i https://metrics.kerfbit.dev/health \
  -H "CF-Access-Client-Id: <id>" \
  -H "CF-Access-Client-Secret: <secret>"
```

Repeat for `mns.kerfbit.dev`, `registry.kerfbit.dev`, and `chat.kerfbit.dev`. Note
that `chat.kerfbit.dev` will show a connection-refused/502 from cloudflared (not an
Access rejection) until `chatbot_api_server` is actually started on 192.168.1.24 —
see "Known gap" below.

On each machine:

```bash
sudo systemctl status cloudflared
cloudflared tunnel info <tunnel-name>
dig metrics.kerfbit.dev   # etc — confirm CNAME resolution
```

Then, from the Android apps (on cellular data): enable the relay switch in Settings,
enter the relay hostname(s) and the service token Client ID/Secret, and confirm chat
and the metrics/mns/registry dashboards load. Finally, back on home Wi-Fi, flip the
relay switch off and confirm LAN-direct mode still works unchanged.

## Operational notes

- **Fastest kill switch** if the tablet is lost or compromised: revoke the Access

  Service Token (Access → Service Auth → Service Tokens → Revoke). This invalidates
  it at Cloudflare's edge instantly, with no access to either home machine required.

- **Secondary cutoff:** `sudo systemctl stop cloudflared` on either machine, delete

  the tunnel, or remove the DNS record — any of these fully closes the public path
  while leaving LAN-direct access unaffected.

- **Rotating a token:** create a new service token, attach it to the Access policy,

  update the Client ID/Secret stored in both Android apps, then revoke the old
  token.

- **Known gap:** `chatbot_api_server` is currently stopped on 192.168.1.24. Wiring

  `chat.kerfbit.dev` does not require starting it, but it won't return real
  responses until it is — don't mistake this for a broken relay.

- The `metrics_api_server` currently running on 192.168.1.16 is a dev-mode

  deployment (user `rodney`, not the hardened `/opt/adai` bundle from
  `install_server_bundle.sh`). The tunnel proxies to `localhost:8081` regardless of
  which deployment mode is running, so this doesn't block the relay — hardening
  that deployment is a separate, pre-existing recommendation.
