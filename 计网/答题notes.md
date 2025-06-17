## 传输层
### TCP 连接/释放

![[Pasted image 20250617110531.png]]
![[Pasted image 20250617110651.png]]
![[Pasted image 20250617110712.png]]
### 拥塞控制
![[Pasted image 20250617111408.png]]
**慢开始算法：cwnd 值从1开始，每收到⼀个ACK，就让 cwnd+1（当 cwnd<ssthress 时适⽤）**
**拥塞避免算法：在⼀个RTT内，即使收到多个ACK，也只能让 cwnd+1（当 cwnd≥ssthress 时适⽤）**
**快重传：当发送⽅收到三个确认号相同的冗余ACK时，⽴即重传对应报⽂段**
**快恢复算法：⼀旦发⽣快重传，就将阈值、cwnd 都设为当前 cwnd 的⼀半，然后切换到为“拥塞避免算法”**

## 网络层
### IP数据报
![[Pasted image 20250617112421.png]]
### IP 路由转发
![[Pasted image 20250617113436.png]]
![[Pasted image 20250617113551.png]]![[Pasted image 20250617113618.png]]
**采用 CIDR 技术后，由于 “路由聚合”，一个IP地址在转发表中可能会匹配多个表项，此时应使用最长前缀匹配原则**
![[Pasted image 20250617113853.png]]
![[Pasted image 20250617113917.png]]
### CIDR地址分配
![[Pasted image 20250617114104.png]]

### NAT（内网-外网）
![[Pasted image 20250617114657.png]]
### DHCP
![[Pasted image 20250617115509.png]]
### ARP

![[Pasted image 20250617115620.png]]