import torch

# Đường dẫn đến tệp .pth của bạn
file_path = '/home/server12gb/Desktop/Bach/UparChallenge_baseline/pipeline/WACV_UPAR_2023-master/Result/pretrained/swin_base_patch4_window7_224_22kto1k.pth'

# Sử dụng torch.load để đọc nội dung của tệp .pth
content = torch.load(file_path)

# In nội dung
if 'model' in content:
    # Nếu 'model' tồn tại trong checkpoint
    print('hehe')
    # Tiếp tục thực hiện các hành động khác ở đây nếu cần thiết
else:
    # Nếu 'model' không tồn tại trong checkpoint
    print("Không tìm thấy khóa 'model' trong checkpoint.")
    # Có thể xử lý theo cách khác tùy thuộc vào yêu cầu của bạn


