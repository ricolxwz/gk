---
title: k8s:架构
comments: true
---

## 基础单元

### pod, service, ingress

k8s的最小单元是pod. pod是容器外的一层抽象, 通常是一些容器组成的一个应用. 每一个pod都有一个ip地址, 注意, 不是每一个容器有一个ip地址, 当pod死亡之后新建的pod会有不同的ip地址. 因此, k8s通过引入一个service的概念, 通过引入一个虚拟的静态ip地址和DNS名称来稳定地暴露这些pod, 即使pod死亡, service所拥有的静态ip地址和DNS名称也不会改变. 有一些service是external service, 另一些是internal service, internal service是一些不会被外界用http访问的服务, 如数据库, external service是一些会被外界用http访问的服务. ingress用于处理http/https流量的路由, 将外部流量根据规则引导到对应的internal service.

### configmap, secret

假设当前你有两个pods, 一个pod是你的app, 一个pod是app的数据库. 通常情况下, 我们会在app里面写死数据库的uri, 比如postgress://...<postgress-db-test>, 但是如果数据库的名称发生改变的话, 我们就要重新build一个镜像, 这样是非常费时费力的, k8s提供了一个configmap, 只要把这个configmap链接到app这个pod上, 然后将db_url指向postgress-db-test, 就可以保证在app镜像内使用的是configmap定义的镜像, 这样就具有动态性. 但是当你发现你要map数据库的用户和密码的时候, 会产生安全问题, 因为configmap是以明文记录的. 这个时候就要用到secret. secret就像configmap, 但是是经过base64编码的. 通过configmap和secret定义的变量就是环境变量, 可以直接使用.

### volumes

如果你要你的数据persist, 可以使用k8s提供的volumes, 这个volumes可以是在k8s集群内, 也可以在k8s集群外, 如云端存储. 当pod重启的时候, 你声明的数据就会persist. 

### deployment

现在, 我们的app的pod死了, 这就意味着我们无法通过https://my-app.com访问, k8s中的service不仅提供静态ip和DNS名称, 还充当了一个负载均衡的作用, 所以如果第一个pod死了, 它会把流量导到另一个replicate, 即另一个副本. 这些pod都属于同一个blueprint, 这个blueprint就叫做deployment. 在k8s中, 你不会去创造pod, 但是你会去创造deployment, 你可以定义多少的replicates. deployment通过replicaset管理同一类型的pod的所有副本. 所以抽象层次为:  deployment -> replicaset -> pod -> container, 每次deployment文件更新的时候, kubernetes会创建新的replicaset, 并逐渐用新的replicaset定义的类型的pod替换旧的replicaset中的pod, 旧的replicaset在完成更新后, 通常不会删除, 而是保留在集群中, 但是不再管理任何pod, 保留的replicaset提供了回滚到之前版本的能力, 因此一个deployment不一定只对应一个replicaset.

### statefulset

在k8s中, 数据库的pod是不能通过deployment创建副本的, 因为数据库是有状态的, 如果我们通过deployment创建了一个数据库的副本, 那么数据库1和数据库2可能同时写入同一个数据库, 导致数据的不一致. k8s提供的statefulset专门用于有状态的pod, 它能够保证几个副本之间保持同步. 在业界, 数据库通常都放在k8s集群的外面, 因为部署一个有状态的pod是相当困难的.

## 架构

### 从节点

在k8s中, 通常将workder node, 即真正跑服务的node称之为node, 节点. 在每一个节点上, 会有3个服务: 必须要有容器的运行时, 如docker, containerd; 还要有kubelet, 主要负责监控和管理节点上的pods, 包括生命周期, 资源管理, 健康检查, 以及和容器运行时和API server的通信. 我们通过service和pods通信, service将流量负载均衡发到各个副本上, 这个流量的分发由kube proxy负责, 这个kube proxy非常智能, 如果节点1上的app发送了一个请求给数据库service, 这个service会通过kube proxy将请求发到相同节点, 节点1上的数据库pod, 而不会发到节点2的数据库pod, 能够减缓网络的拥塞.

### 主节点

Master nodes, 主节点. 在每个主节点上, 会有4个服务: API server, 它是集群的gateway, 当有请求进来的时候, 它会处理请求, 然后响应, 可以通过多种方式向API server发起请求, 如UI, API CLI(kubectl). 如果你向API server发起新建一个pod的请求, 这个请求在通过API server后会交给scheduler处理, scheduler会智能决定将pod放在哪一个节点上. 注意, scheduler只是决定将pod放在哪一个节点上, 真正执行创建pod指令的是那个节点上的kubelet. 第三个服务是controller manager, 它会检测pod的健康状态, 并通过创建replicate恢复到健康状态, 它会像scheduler发送请求, 让对应节点的kubelet新建一个pod. 最后一个服务是etcd, 它是一个key-value的存储服务, 当集群的状态发生变化的时候, 会更新etcd中的键值对, API server, scheduler, controller manager会从etcd拿去集群的状态, 相当于etcd是大脑. 通常情况下, 主节点可能会有多个, 会通过负载均衡将请求发到不同主节点的API server. 当然, etcd会被分布式存储.