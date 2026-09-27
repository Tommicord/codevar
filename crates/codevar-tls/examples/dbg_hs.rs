//! Debug: trace an in-memory TLS1.3 handshake.

use codevar_tls::*;
use std::sync::Arc;

const LEAF_PEM: &[u8] = b"-----BEGIN CERTIFICATE-----
MIIBuDCCAV+gAwIBAgIUI45Sxo+yHGLkPe4tXHxugB3Lhu4wCgYIKoZIzj0EAwIw
FjEUMBIGA1UEAwwLZXhhbXBsZS5jb20wHhcNMjYwOTIzMjAyMTQxWhcNNDYwOTE4
MjAyMTQxWjAWMRQwEgYDVQQDDAtleGFtcGxlLmNvbTBZMBMGByqGSM49AgEGCCqG
SM49AwEHA0IABL5z8IGhXvq7uF8xXipWmy6wpf9TtmwjEARYe+DrRZe7N8Nx/pYg
tOads0U1R/xzZvbzPrkiSCo+dZ9b0nKDvC6jgYowgYcwHQYDVR0OBBYEFIlrOHUy
sFOYDyIQlSqE41PLG/0ZMB8GA1UdIwQYMBaAFIlrOHUysFOYDyIQlSqE41PLG/0Z
MA8GA1UdEwEB/wQFMAMBAf8wNAYDVR0RBC0wK4ILZXhhbXBsZS5jb22CD3d3dy5l
eGFtcGxlLmNvbYILKi53aWxkLnRlc3QwCgYIKoZIzj0EAwIDRwAwRAIgUxJKwkme
Vz6RuQm949/phJHV5pb7ghO7S6Yk1fqvHSUCIAdmL88XANgTFjKfEgg3U890l+OD
sJf8oOZ6rFyNdNBS
-----END CERTIFICATE-----
";
const LEAF_KEY_PEM: &[u8] = b"-----BEGIN PRIVATE KEY-----
MIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQgk1vhLdt3cRaaH+fX
2hjwNQNzwnpoCdRcRaYF2WHgxy+hRANCAAS+c/CBoV76u7hfMV4qVpsusKX/U7Zs
IxAEWHvg60WXuzfDcf6WILTmnbNFNUf8c2b28z65IkgqPnWfW9Jyg7wu
-----END PRIVATE KEY-----
";

fn main() {
    let client = TlsClientConnection::new(
        ClientConfig::dangerous_insecure(),
        ServerName::try_from_str("example.com").unwrap(),
    )
    .unwrap();
    let server = TlsServerConnection::new(ServerConfig::new(
        CertifiedKey::from_pem(LEAF_PEM, LEAF_KEY_PEM).unwrap(),
    ));
    let (mut c, mut s) = (client, server);
    for i in 0..10 {
        let fc = c.write_tls();
        let fs = s.write_tls();
        if !fc.is_empty() { s.read_tls(&fc).unwrap(); }
        if !fs.is_empty() { c.read_tls(&fs).unwrap(); }
        let sc = s.process_new_packets();
        let cc = c.process_new_packets();
        println!(
            "round {i}: moved c={} s={}, after: cw={} sw={} hs c={} s={}",
            fc.len(), fs.len(), c.wants_write(), s.wants_write(),
            c.is_handshaking(), s.is_handshaking()
        );
        if let Err(e) = cc { println!("  client err: {e:?}"); }
        if let Err(e) = sc { println!("  server err: {e:?}"); }
        if !c.is_handshaking() && !s.is_handshaking() { println!("done"); break; }
        let progressed = !fc.is_empty() || !fs.is_empty() || c.wants_write() || s.wants_write();
        if !progressed { println!("stalled"); break; }
    }
}
