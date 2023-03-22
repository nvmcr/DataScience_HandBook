# Cloud Essentials
## Table of Contents
1. [Intro](#Intro)
    1. [Client-Server Model](#client-server-model)
    2. [Cloud Computing](#cloud-computing)
2. [Elastic Compute Cloud (EC2)](#EC2)
    1. [EC2 Instance Types](#types-of-ec2-instances)
    2. [Pricing](#pricing)
    3. [Scaling](#auto-scaling)
    4. [Elastic Load Balancing](#elastic-loa--ba-ancing)
    5. [Messaging and Queuing](#Messaging-and-queuing)
    6. [Other services](#Quick-bites-of-other-services)
3. [Networking](#Networking)
4. [Storage and Databases](#Storage-and-Databases)
## Intro
The key concept of Amazon Web Services (AWS) is that *only pay for what is used*. The best thing about cloud when compared to on-premises data centers is to get as many resources as needed at any time and no need to get rid of them when not needed, that way one will pay only for what one used.
### Client-Server Model
A client can be a web browser or an application that a customer interacts to make requests to computer servers. One such type of virtual server is Amazon Elastic Compute Cloud(EC2). What about API?
> We can simplify the concept with an analogy of a restaurant. In this analogy, the customer is like a client, as they are the ones who are making a request for a service or product. The waiter is like an API, as they act as an intermediary between the customer and the chef, taking the customer's order and delivering it to the chef. The chef is like a server, as they provide the resources and perform the necessary work to prepare and deliver the order to the customer.
### Cloud Computing
The on-demand delivery of IT resources over the internet with pay-as-you-go pricing. Types of clud computing include: Infrastructure as a Service (IaaS), Platform as a Service (PaaS) and Software as a Service (SaaS).
* IaaS is a cloud computing model that provides customers with access to virtualized computing resources, such as servers, storage, and networking. With IaaS, the customer is responsible for managing the operating system, middleware, and applications. Examples of IaaS providers include Amazon Web Services (AWS), Microsoft Azure, and Google Cloud Platform.
* PaaS is a cloud computing model that provides customers with a platform on which they can develop, run, and manage their own applications. With PaaS, the customer is responsible for developing and managing their applications, while the provider takes care of the underlying infrastructure, such as servers, storage, and networking. Examples of PaaS providers include Heroku, Google App Engine, and Microsoft Azure.
* SaaS is a cloud computing model that provides customers with access to software applications that are hosted and managed by a third-party provider. With SaaS, the customer only needs to access the application through a web browser or mobile app, and the provider is responsible for managing the infrastructure, middleware, and application. Examples of SaaS providers include Salesforce, Dropbox, and Google Apps.
The amount of control the customer has decreases from IaaS to SaaS.
## EC2
> A service that one can use to gain access to virtual servers is called Elastic Compute Cloud (EC2).

AWS EC2 is an amazing alternative to on-premises data centers. AWS has servers, racked and stacked them and they are already online ready to use. EC2 runs on top of physical host machines managed by AWS using **virtualization**. When one uses EC2 instance, they are not taking an entire host. Instead, one is sharing the host with multiple instances/virtual machines. A *hypervisor* running on host machine is responsible for sharing the resources between virtual machines. This is called **multilatency**. Each instance is isoloated from one another. 
While using EC2 instance, one can choose the operating system (Windows or Linux). One can also configure what software to run on that instance like different web aps, databases. User can increase memory and compute for a particular instance after realizing that the application is starting to max out that instance. This is called **vertical scaling**. The user can also control the netwrok aspect of the instance.
### Types of EC2 Instances
There are different types of EC2 instances are grouped  under an instance familty and are optimized for different tasks. 
1. General Purpose Instances 
    They provide a **balance** of compute, memory and networking tasks. They can be used for variety of workloads like application servers, gaming servers, small and medium databases, etc. They are chosen if the user doesn't need optimization in a single resource area.
2. Compute Optimized Instances 
    They are ideal for compute bound applications taht benefit from **high-perfomance** processors. Ideal for high perfomance web servers, compute-intensive applications servers adn also for batch processing workloads that require many transanctions in a single group.
3. Memory Optimized Instances 
    They are designed to deliver fast performance for workloads that process **large datasets in memory**. If user has a workload that require large amounts of data to be preloaded before running the application, memory optimized instances are useful.
4. Acclearated Compuing Instances 
    These use **hardware accelarators** to perform certain function more efficiently like floating point number calculations, graphics processing etc.
5. Storage Optimized Instances 
    They are designed for workloads that require **high, sequential read and write access to large datasets** on local storage. Examples include data warehousing applications, high-frequency online transaction processing systems etc. These servers provides low latency and high input/output operations per second (IOPS).
### Pricing
AWS offers 5 types of pricing.
1. On-Demand: Pay for the duration an instance runs. No commitments or contracts.
2. Savings Plan: Commitment to aonsistent amount of usage measured in dollars per hour for one.three year term.
3. Reserved Instances: Suited for steady-state workloads or ones with predictable usage.
4. Spot Instances: This plan allows user to use spare instances but AWS can claim the instance at any time.
5. Dedicated Hosts: Physical hosts dedicated for user. Nobody else will share tenancy of that host.
### Auto Scaling
>Scalability means beginning with the only resources the user need and designing architecture to automatically respond to changing demand by scaling out or in.
**Amazon EC2 Auto Scaling** is the services that provides the scaling process for EC2 instances. There are two approaches to use auto scaling:
* Dynamic Scaling: Responding to changing demand
* Predictive Scaling: Automatically schedule the right number of instances based on predicted demand. 

There are several configurations that can be set for an auto scaling group. The user must set the minimum number of instances, desired capacity (if None, desire = minimum) and maximum capacity (how much to scale during increase in demand).
### Elastic Load Balancing
We use restaurant analogy again. Say customers(requests) flow suddenly increased. With the help of auto scaling we increased our cooks (instances) but how does the waiter(api) know which cook the order should go to. What if most of the orders go to cook 1, leaving others without work. To manage the requests load, we use a load balancer that ensures there is even distribution of workload. AWS offers Elastic Load Balancing (ELB).
> Elastic Load Balancing automatically distributes incoming application traffic across multiple resources/insatnces.

ELB is a **regional construct** and it scales automatically. As the traffic groes, ELB handles additional throughput. Also when the extra insatnces are available, the auto scaling service lets the ELB know about extra resources and ELB distributes accordingly. Once the throughput/requests decreases, ELB will stop new input into few instances and once those instances are out of requests, they are terminated. 
Apart from handling such external traffic, it handles ordering tier(front end) and production tier(back end). Say if we got a new instance in back end, then the insatnce has to let every insatnce of front end know that its ready and the front end resources has to change distribution of every backend instances they are connected with. But with ELB, as it is regional, every front end instance come to ELB (single url for all front end instances) and ELB distributes to available back end instances. Now if a new instance is available in backend, then it will only let ELB know and ELB will manage the load. The front end doesn't need to know anything about whats happening in back end.
### Messaging and queuing
In the restaurant case, the service flow works as long as the waiter and cook is in sync. What if cook is busy with an order but waiter is waiting for the cook to take another order. After a certain time the waiter might drop that order and go for a new customer order. To handle this we can have something like a board where waiter lists everything that the cook has to do which is nothing but we are placing the orders in a buffer. Just like waiter and cook, applications might be facing issues with transferring messages. If application B fails to take the message, then application A also fails. This is called *tighlty coupled* architecture or **monolithic**. AWS uses *lossely  coupled* architecture or **microservices**. In this even if one component fails, it won't cause cascading failures. We will have a message queue where application A sends all the messages. In case of B failure, the messages just get piled up in buffer and gets transmissiioned once B is back. To achieve this AWS uses two services called Amazon **Simple Queue Service**(SQS) and **Simple Notification Service**(SNS). 
SQS allows to send, store and receive messages between software components at any volume. The messages are placed in SQS queues. On the other hand, SNS sends messages to serices like notifications to end users. It has a publisher/subscriber (pub/sub) model. We can create a SNS topic which is a channel for messages to be delivered. Then the configure subscribers(end users) to that topic and we publish those messages to subscribers. Thus sending one message to all the subscribers.
### Quick bites of other services
Though most of things EC2 does are automatic, we have to do EC2 setups, managing instances, patching instances with new software packages, etc. To decrease such extra tasks, AWS offers multiple serverless options.
> Serverless means that we cannot see/access the underlying infrastructure or instances that are hosting the applicaion.

Everything is taken care by AWS. One such serverless comput eoption is **AWS Lambda**. We upload our code into a Lambda function and configure a trigger. The AWS service waits for the trigger and when there is one, the code is run automatically in an environment that is taken care by AWS. lambda is designed to run code *under 15 minutes*. So it can't run deep learning tasks but most suitable for web backend, handling requests that takes less than 15 minutes to complete. 
If we need to access the environment but still want the efficieny and portability, we cna use AWS container services like Amazon **Elastic Container Service** (ECS) and **Elastic Kubernetes Service** (EKS). Both these services are docker container orchestration tools.
> A container is a package of our code where we pack our application, dependencies and configurations that it needs.

Think it like a conda or pip environment that has details needed to replicate our system. These containers run on top of EC2 instance. There are multiple docker containers that run in isolation of each other. We need to start, stop, restart and monitor these multiple containers called clusters. This process of tasks is called container orchestration. ECS and EKS are designed to help these orchestrations. ECS helps to run containerized applications without need of our own orchestration software. EKS also does similar with different tools. 
But again these two services run over EC2 instance. We need to go serverless, then AWS offers **Fargate**. It is a serverless compute platform for ECS and EKS. In general, say we need to host applications and want access to that underlying os like windows, linux then we have to go for EC2 instance. But we wawnt to host some short running functions or even-driven and we dont need to knw about underlying environment, we go for AWS lambda. Similarly we choose ECS and EKS and then choose to go with EC2 or Fargate.
> Difference between docker and kubernetes is, docker is foundation of containerization and provides a way to package and deploy applications, while Kubernetes builds on top of Docker and provides tools for managing and orchestrating multiple containers across multiple hosts.
## Networking
As there are millions of customers who use AWS services and vast number of resources customers created, there should be boundaries around resources such a way that network traffic would be able to flow between them unrestricted. AWS offers **Virtual Private Cloud**(VPC) to establish boundaries around AWS resources. Amazon VPC enables us an isolated section of AWS cloud. In that section, we organize our resources into subnets. To allow public traffic from internet to access our VPC, we attach an **internet gateway** to the VPC. Similarly we have only private resources in VPC, to grant access to required users, we use a **Virtual Private Network**(VPN). VPN uses same path as VPC except our traffic is encrypted. As we use the same connection as VPC, there might be slow downs. So AWS offers one more service **Direct Connect**. It provides a dedicated connection to our VPC.  This helps us to reduce network costs and increase the bandwidth.
![AWS Direct Cloud](direct_connect.png)
What happens when a customer clicks on our application say a website? 

Once a customer reuests data drom application, Amazon **Route 53** uses DNS resolution which gets IP address through the domain name and then the customer request is sent to nearest edge location (a data centre string cache data for low latency) through Amazon **CloudFront**. The Amazn CloudFront connects to the appliction load balancer which sends the incoming packets to EC2 instance.
## Storage and Databases
