# 运行100次，每次传递不同的 random_state（从0到99）
for ($i=0; $i -lt 1000; $i++) {
    python SNS.py $i | Out-File -Append output.log
}