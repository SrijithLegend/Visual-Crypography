from Crypto.PublicKey import ECC
import Crypto.Signature.DSS
from Crypto.Hash import SHA256
import numpy as np
from PIL import Image
import qrcode


mykey = ECC.generate(curve='p256')

private_key_pem = mykey.export_key(format='PEM')
public_key_pem = mykey.public_key().export_key(format='PEM')

hash_mykey = SHA256.new(private_key_pem.encode())

signer = Crypto.Signature.DSS.new(mykey, 'fips-186-3')
signature = signer.sign(hash_mykey)
qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_L,
    box_size=1,
    border=0,
)
qr.add_data(signature)
qr.make(fit=True)

qr_img = qr.make_image(fill_color="black", back_color="white").convert("1")
qr_array = np.array(qr_img)

height, width = qr_array.shape

share1 = np.zeros((height * 2, width * 2), dtype=bool)
share2 = np.zeros((height * 2, width * 2), dtype=bool)

block_white_0 = np.array([[True, False], [False, True]])
block_white_1 = np.array([[False, True], [True, False]])

for r in range(height):
    for c in range(width):
        is_black = not qr_array[r, c]
        rand_choice = np.random.rand() > 0.5
        
        if is_black:
            s1_sub = block_white_0 if rand_choice else block_white_1
            s2_sub = ~s1_sub
        else:
            s1_sub = block_white_0 if rand_choice else block_white_1
            s2_sub = s1_sub.copy()
            
        share1[r*2:(r+1)*2, c*2:(c+1)*2] = s1_sub
        share2[r*2:(r+1)*2, c*2:(c+1)*2] = s2_sub

img_share1 = Image.fromarray(~share1)
img_share2 = Image.fromarray(~share2)

img_share1.save("share1.png")
img_share2.save("share2.png")

overlay = np.bitwise_or(share1, share2)
reconstructed_img = Image.fromarray(~overlay)
reconstructed_img.save("reconstructed_qr.png")