import os
import gdown

if os.path.exists('koT5_summary_last/config.json') and os.path.exists('koT5_summary_last/pytorch_model.bin') :
    print("== Data existed.==")
    pass
else:
    os.system("rm -rf koT5_summary_last")
    os.system("mkdir koT5_summary_last")
    file_id = "1GMFEw3uhvpyzL1c6zS7-q4pJmFphV6rE"
    output = './koT5_summary_last/config.json'
    print("Download config.json")
    gdown.download(id=file_id, output=output, quiet=False)

    file_id = "12wb8sdD3zOQ_NHzHS0FPze5RxbVNhs_G"
    output = './koT5_summary_last/pytorch_model.bin'
    print("Download pytorch_model.bin")
    gdown.download(id=file_id, output=output, quiet=False)