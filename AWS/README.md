# Cloud Essentials
## Table of Contents
1. [Intro](#Intro)
    1. [Client-Server Model](#client-server-model)
    2. [Cloud Computing](#cloud-computing)
2. [Elastic Compute Cloud (EC2)](#EC2)
    1. [EC2 Instance Types](#types-of-ec2-instances)
    2. [Pricing](#pricing)
    3. [Scaling](#auto-scaling)
    4. [Elastic Load Balancing](#elastic-load-balancing)
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
    These use hardware accelarators to perform certain function more efficiently like floating point number calculations, graphics processing etc.
5. Storage Optimized Instances
    They are designed for workloads that require high, sequential read and write access to large datasets on local storage. Examples include data warehousing applications, high-frequency online transaction processing systems etc. These servers provides low latency and high input/output operations per second (IOPS).
### Pricing
AWS offers 5 types of pricing.
1. On-Demand: Pay for the duration an instance runs. No commitments or contracts.
2. Savings Plan: Commitment to aonsistent amount of usage measured in dollars per hour for one.three year term.
3. Reserved Instances: Suited for steady-state workloads or ones with predictable usage.
4. Spot Instances: This plan allows user to use spare instances but AWS can claim the instance at any time.
5. Dedicated Hosts: Physical hosts dedicated for user. Nobody else will share tenancy of that host.
### Auto Scaling
>Scalability means beginnin with the only resources the user need and designing architecture to automatically respond to changing demand by scaling out or in.
**Amazon EC2 Auto Scaling** is the services that provides the scaling process for EC2 instances. There are two approaches to use auto scaling:
* Dynamic Scaling: Responding to changing demand
* Predictive Scaling: Automatically schedule the right number of instances based on predicted demand.
There are several configurations that can be set for an auo scaling group. The user must set the minimum number of instances, desired capacity (if None, desire = minimum) and maximum capacity (how much to scale during increase in demand).
### Elastic Load Balancing
