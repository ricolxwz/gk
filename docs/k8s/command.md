---
title: k8s:命令
comments: true
---

## 使用`kubectl`命令

```
$ kubectl create deployment nginx-depl --image=nginx # 创建了一个只有一个nginx容器的pod, 并且只有一个副本. 这个命令会自动生成一个YAML配置文件
deployment.apps/nginx-depl created
$ kubectl get pod
NAME                          READY   STATUS              RESTARTS   AGE # 这里的7fb9f是replicaset下一个副本pod的id
nginx-depl-6777bffb6f-7fb9f   0/1     containercreating   0          0s
$ kubectl get replicaset # 这里的6777bffb6f是replicaset的id
name                    desired   current   ready   age
nginx-depl-6777bffb6f   1         1         0       0s
$ kubectl edit deployment nginx-depl # 可以修改YAML配置文件, 如果修改了, 就会产生另一个replicaset
$ kubectl logs nginx-depl-6777bffb6f-7fb9f
/docker-entrypoint.sh: /docker-entrypoint.d/ is not empty, will attempt to perform configuration
/docker-entrypoint.sh: Looking for shell scripts in /docker-entrypoint.d/
/docker-entrypoint.sh: Launching /docker-entrypoint.d/10-listen-on-ipv6-by-default.sh
10-listen-on-ipv6-by-default.sh: info: Getting the checksum of /etc/nginx/conf.d/default.conf
10-listen-on-ipv6-by-default.sh: info: Enabled listen on IPv6 in /etc/nginx/conf.d/default.conf
/docker-entrypoint.sh: Sourcing /docker-entrypoint.d/15-local-resolvers.envsh
/docker-entrypoint.sh: Launching /docker-entrypoint.d/20-envsubst-on-templates.sh
/docker-entrypoint.sh: Launching /docker-entrypoint.d/30-tune-worker-processes.sh
/docker-entrypoint.sh: Configuration complete; ready for start up
2024/09/15 23:35:47 [notice] 1#1: using the "epoll" event method
2024/09/15 23:35:47 [notice] 1#1: nginx/1.27.1
2024/09/15 23:35:47 [notice] 1#1: built by gcc 12.2.0 (Debian 12.2.0-14)
2024/09/15 23:35:47 [notice] 1#1: OS: Linux 6.10.7-orbstack-00280-gd3b7ec68d3d4
2024/09/15 23:35:47 [notice] 1#1: getrlimit(RLIMIT_NOFILE): 1048576:1048576
2024/09/15 23:35:47 [notice] 1#1: start worker processes
2024/09/15 23:35:47 [notice] 1#1: start worker process 29
2024/09/15 23:35:47 [notice] 1#1: start worker process 30
2024/09/15 23:35:47 [notice] 1#1: start worker process 31
2024/09/15 23:35:47 [notice] 1#1: start worker process 32
2024/09/15 23:35:47 [notice] 1#1: start worker process 33
2024/09/15 23:35:47 [notice] 1#1: start worker process 34
2024/09/15 23:35:47 [notice] 1#1: start worker process 35
2024/09/15 23:35:47 [notice] 1#1: start worker process 36
$ kubectl describe pod nginx-depl-6777bffb6f-7fb9f
Name:             nginx-depl-6777bffb6f-7fb9f
Namespace:        default
Priority:         0
Service Account:  default
Node:             orbstack/198.19.249.2
Start Time:       Mon, 16 Sep 2024 09:35:32 +1000
Labels:           app=nginx-depl
                  pod-template-hash=6777bffb6f
Annotations:      <none>
Status:           Running
IP:               192.168.194.10
IPs:
  IP:           192.168.194.10
  IP:           fd07:b51a:cc66:a::a
Controlled By:  ReplicaSet/nginx-depl-6777bffb6f
Containers:
  nginx:
    Container ID:   docker://8cbf8e6aa3abbf58553295f0c03ce534c58ab52590571a3cbac94d960cd04047
    Image:          nginx
    Image ID:       docker-pullable://nginx@sha256:04ba374043ccd2fc5c593885c0eacddebabd5ca375f9323666f28dfd5a9710e3
    Port:           <none>
    Host Port:      <none>
    State:          Running
      Started:      Mon, 16 Sep 2024 09:35:47 +1000
    Ready:          True
    Restart Count:  0
    Environment:    <none>
    Mounts:
      /var/run/secrets/kubernetes.io/serviceaccount from kube-api-access-2jgsl (ro)
Conditions:
  Type                        Status
  PodReadyToStartContainers   True
  Initialized                 True
  Ready                       True
  ContainersReady             True
  PodScheduled                True
Volumes:
  kube-api-access-2jgsl:
    Type:                    Projected (a volume that contains injected data from multiple sources)
    TokenExpirationSeconds:  3607
    ConfigMapName:           kube-root-ca.crt
    ConfigMapOptional:       <nil>
    DownwardAPI:             true
QoS Class:                   BestEffort
Node-Selectors:              <none>
Tolerations:                 node.kubernetes.io/not-ready:NoExecute op=Exists for 300s
                             node.kubernetes.io/unreachable:NoExecute op=Exists for 300s
Events:
  Type    Reason     Age   From               Message
  ----    ------     ----  ----               -------
  Normal  Scheduled  10m   default-scheduler  Successfully assigned default/nginx-depl-6777bffb6f-7fb9f to orbstack
  Normal  Pulling    10m   kubelet            Pulling image "nginx"
  Normal  Pulled     10m   kubelet            Successfully pulled image "nginx" in 13.729s (13.729s including waiting)
  Normal  Created    10m   kubelet            Created container nginx
  Normal  Started    10m   kubelet            Started container nginx
$ kubectl exec -it nginx-depl-6777bffb6f-7fb9f -- /bin/bash
$ kubectl get deployment
NAME         READY   UP-TO-DATE   AVAILABLE   AGE
nginx-depl   1/1     1            1           13m
$ kubectl delete deployment nginx-depl # 删除掉所有的replicaset和所有的对应pod
```

## 使用YAML配置文件

`kubectl apply -f <xxx.yaml>`会执行 YAML 文件中定义的 deployment.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx-depl
  labels:
    app: nginx-depl # deployment本身的标签
spec:
  replicas: 1
  selector:
    matchLabels:
      app: nginx # deployment选择器选择的pod的标签
  template:
    metadata:
      labels:
        app: nginx # 新创建的pod的标签
    spec:
      containers:
        - name: nginx
          image: nginx:1.16
          ports:
            - containerPort: 8080 # 可省略
```

```
$ kubectl apply -f nginx-depl.yaml
deployment.apps/nginx-depl created
$ kubectl get pod
NAME                          READY   STATUS    RESTARTS   AGE
nginx-depl-7b965f675d-k4p65   1/1     Running   0          5s
$ # 修改replicas为2
$ kubectl apply -f test.yaml # 注意到这里的replicaset的id没变, 因为只是改变了副本的数量
deployment.apps/nginx-depl configured
$ kubectl get pod
NAME                          READY   STATUS    RESTARTS   AGE
nginx-depl-7b965f675d-k4p65   1/1     Running   0          63s
nginx-depl-7b965f675d-kfl5x   1/1     Running   0          20s
$ kubectl get replicaset
NAME                    DESIRED   CURRENT   READY   AGE
nginx-depl-7b965f675d   2         2         2       72s
```

每个 deployment 的配置应该有 3 个部分, metadata, specification, 和 status, k8s 会比较 specification 的状态和 status 之间的区别, 然后会尝试达到 specification 的状态, 这个 staus 部分不是在`nginx-depl.yaml`文件中, 而是从 etcd 动态获取的. 我们的`nginx-depl.yaml`文件只是声明了其中的metadata和specification部分, 可以通过`kubectl get deployment nginx-depl -o yaml > nginx-depl-full.yaml`查看所有的配置.

service 的配置可以参考:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: nginx-service
spec:
  selector:
    app: nginx # 选择pod的标签
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8080
```

```
$ kubectl apply -f nginx-service.yaml
service/nginx-service created
$ kubectl get service
NAME            TYPE        CLUSTER-IP        EXTERNAL-IP   PORT(S)   AGE
kubernetes      ClusterIP   192.168.194.129   <none>        443/TCP   172m
nginx-service   ClusterIP   192.168.194.169   <none>        80/TCP    20s
$ kubectl describe service nginx-service
Name:              nginx-service
Namespace:         default
Labels:            <none>
Annotations:       <none>
Selector:          app=nginx
Type:              ClusterIP
IP Family Policy:  SingleStack
IP Families:       IPv4
IP:                192.168.194.169
IPs:               192.168.194.169
Port:              <unset>  80/TCP
TargetPort:        8080/TCP
Endpoints:         192.168.194.13:8080,192.168.194.14:8080 # 这些就是被选中的pod的ip
Session Affinity:  None
Events:            <none>
$ kubectl get pod -o wide # 查看pod的ip验证
NAME                          READY   STATUS    RESTARTS   AGE     IP               NODE       NOMINATED NODE   READINESS GATES
nginx-depl-7b965f675d-k4p65   1/1     Running   0          4m47s   192.168.194.13   orbstack   <none>           <none>
nginx-depl-7b965f675d-kfl5x   1/1     Running   0          4m4s    192.168.194.14   orbstack   <none>           <none>
```

最终, 删除deployment和service:

```
$ kubectl delete -f nginx-depl.yaml
deployment.apps "nginx-depl" deleted
$ kubectl delete -f nginx-service.yaml
service "nginx-service" deleted
```

???+ Tip "Tip"

    `containerPort`用来描述pod内部监听的端口, 但是不会改变pod内部实际监听的端口, 如pod内部nginx服务监听的是80端口, `containerPort: 8080`, 不会改变nginx监听的是80的事实. 这主要起到一个告知的作用, 删去`containerPort`也不会发生什么. `targetPort`用于service中, 它用于将流量定向到pod的正确端口, 如果你设置了`targetPort: 8080`, 那么流量就会定向到pod的`8080`端口. 如果省略`targetPort`, 会默认使用`port`的值, 也就是service本身暴露的端口作为目标端口.