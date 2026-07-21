from Crypto.PublicKey import ECC


# Generate a new ECC key pair using the P-256 curve
mykey = ECC.generate(curve='p256')
print('mykey:', mykey)

# Export the private key in PEM format
private_key_pem = mykey.export_key(format='PEM')
print('Private Key (PEM):', private_key_pem)

# Export the public key in PEM format
public_key_pem = mykey.public_key().export_key(format='PEM')
print('Public Key (PEM):', public_key_pem)