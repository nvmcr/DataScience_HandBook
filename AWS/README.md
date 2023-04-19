# Cloud Essentials
<details>
<summary>Table of Contents</summary>

## Table of Contents
1. [Intro](#Intro)
    1. [Client-Server Model](#client-server-model)
    2. [Cloud Computing](#cloud-computing)
2. [Elastic Compute Cloud (EC2)](#EC2)
    1. [EC2 Instance Types](#types-of-ec2-instances)
    2. [Pricing](#pricing)
    3. [Scaling](#auto-scaling)
    4. [Elastic Load Balancing](#elastic-load-balancing)
    5. [Messaging and Queuing](#Messaging-and-queuing)
    6. [Other Services](#Other-services)
        1. [Lambda](#Lambda)
        2. [ECS and EKS](#ecs-and-eks)
        3. [Fargate](#Fargate)
            1. [Docker vs Kubernetes](#Docker-vs-Kubernetes)
3. [Infrastructure](#Infrastructure)
    1. [Regions](#regions)
    2. [Availability Zones](az)
    3. [Edge Location and Outposts](#edge-locations)
5. [Networking](#Networking)
6. [Storage and Databases](#Storage-and-Databases)
    1. [EBS](#elastic-block-store-ebs)
    2. [S3](#simple-storage-service-s3)
        1. [EBS vs S3](#EBS-vs-S3)
    3. [EFS](#Elastic-File-System-efs)
        1. [EFS vs EBS](#EFS-vs-EBS)
    4. [RDS](#Relational-Database-Service-rds)
    5. [DynamoDB](#DynamoDB)
    7. [Redshift](#Redshift)
        1. [Database vs Data Warehouse vs Data Lake](#Database-vs-Data-Warehouse-vs-Data-Lake)
    9. [Other Services](#additional-database-services)
        1. [DocumentDB](#documentdb)
        2. [Neptune](#neptune)
        3. [Quantum Ledger Database](#quantum-ledger-database)
        4. [Managed Blockchain](#managed-blockchain)
        6. [DynamoDB Accelerator](#dynamodb-accelerator)

</details>

## Intro
The key concept of Amazon Web Services (AWS) is that *only pay for what is used*. The best thing about cloud when compared to on-premises data centers is to get as many resources as needed at any time and no need to get rid of them when not needed, that way one will pay only for what one used.
### Client-Server Model
A client can be a web browser or an application that a customer interacts to make requests to computer servers. One such type of virtual server is Amazon Elastic Compute Cloud(EC2). What about API?
> We can simplify the concept with an analogy of a restaurant. In this analogy, the customer is like a client, as they are the ones who are making a request for a service or product. The waiter is like an API, as they act as an intermediary between the customer and the chef, taking the customer's order and delivering it to the chef. The chef is like a server, as they provide the resources and perform the necessary work to prepare and deliver the order to the customer.
### Cloud Computing
The on-demand delivery of IT resources over the internet with pay-as-you-go pricing. Types of cloud computing include: Infrastructure as a Service (IaaS), Platform as a Service (PaaS) and Software as a Service (SaaS).
* IaaS is a cloud computing model that provides customers with access to virtualized computing resources, such as servers, storage, and networking. With IaaS, the customer is responsible for managing the operating system, middleware, and applications. Examples of IaaS providers include Amazon Web Services (AWS), Microsoft Azure, and Google Cloud Platform.
* PaaS is a cloud computing model that provides customers with a platform on which they can develop, run, and manage their own applications. With PaaS, the customer is responsible for developing and managing their applications, while the provider takes care of the underlying infrastructure, such as servers, storage, and networking. Examples of PaaS providers include Heroku, Google App Engine, and Microsoft Azure.
* SaaS is a cloud computing model that provides customers with access to software applications that are hosted and managed by a third-party provider. With SaaS, the customer only needs to access the application through a web browser or mobile app, and the provider is responsible for managing the infrastructure, middleware, and application. Examples of SaaS providers include Salesforce, Dropbox, and Google Apps.

The amount of control the customer decreases from IaaS to SaaS. If there is a confusion between IaaS and PaaS, read through this example. Say we need to deploy a web application on the cloud. If we choose IaaS, we will have access to servers, storage and other compute and network services. We ourselves have to install and configure operating system, any other required software etc. But we choose PaaS, along with compute, we will also have required softwares and operating systems pre-built for us. We only need to manage the application. But we can only build applications that is supported by the platform.
## EC2
> A service that one can use to gain access to virtual servers is called Elastic Compute Cloud (EC2).

AWS EC2 is an amazing alternative to on-premises data centers. AWS has servers, racked and stacked them and they are already online ready to use. EC2 runs on top of physical host machines managed by AWS using **virtualization**. When one uses EC2 instance, they are not taking an entire host. Instead, one is sharing the host with multiple instances/virtual machines. A *hypervisor* running on host machine is responsible for sharing the resources between virtual machines. This is called **multilatency**. Each instance is isoloated from one another. 
While using EC2 instance, one can choose the operating system (Windows or Linux). One can also configure what software to run on that instance like different web aps, databases. User can increase memory and compute for a particular instance after realizing that the application is starting to max out that instance. This is called **vertical scaling**. The user can also control the netwrok aspect of the instance.
### Types of EC2 Instances
There are different types of EC2 instances are grouped  under an instance familty and are optimized for different tasks. 
1. General Purpose Instances 
    They provide a **balance** of compute, memory and networking tasks. They can be used for variety of workloads like application servers, gaming servers, small and medium databases, etc. They are chosen if the user doesn't need optimization in a single resource area.
2. Compute Optimized Instances 
    They are ideal for compute bound applications taht benefit from **high-perfomance** processors. Ideal for high perfomance web servers, compute-intensive applications servers and also for batch processing workloads that require many transanctions in a single group.
3. Memory Optimized Instances 
    They are designed to deliver fast performance for workloads that process **large datasets in memory**. If user has a workload that require large amounts of data to be preloaded before running the application, memory optimized instances are useful.
4. Acclearated Compuing Instances 
    These use **hardware accelarators** to perform certain function more efficiently like floating point number calculations, graphics processing etc.
5. Storage Optimized Instances 
    They are designed for workloads that require **high, sequential read and write access to large datasets** on local storage. Examples include data warehousing applications, high-frequency online transaction processing systems etc. These servers provides low latency and high input/output operations per second (IOPS).
### Pricing
AWS offers 5 types of pricing.
1. On-Demand: Pay for the duration an instance runs. No commitments or contracts.
2. Savings Plan: Commitment to a consistent amount of usage measured in dollars per hour for one/three year term.
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
### Other Services
Though most of things EC2 does are automatic, we have to do EC2 setups, managing instances, patching instances with new software packages, etc. To decrease such extra tasks, AWS offers multiple serverless options.
> Serverless means that we cannot see/access the underlying infrastructure or instances that are hosting the applicaion.
#### Lambda
Everything is taken care by AWS. One such serverless comput eoption is **AWS Lambda**. We upload our code into a Lambda function and configure a trigger. The AWS service waits for the trigger and when there is one, the code is run automatically in an environment that is taken care by AWS. lambda is designed to run code *under 15 minutes*. So it can't run deep learning tasks but most suitable for web backend, handling requests that takes less than 15 minutes to complete. 
#### ECS and EKS
If we need to access the environment but still want the efficieny and portability, we cna use AWS container services like Amazon **Elastic Container Service** (ECS) and **Elastic Kubernetes Service** (EKS). Both these services are docker container orchestration tools.
> A container is a package of our code where we pack our application, dependencies and configurations that it needs.

Think it like a conda or pip environment that has details needed to replicate our system. These containers run on top of EC2 instance. There are multiple docker containers that run in isolation of each other. We need to start, stop, restart and monitor these multiple containers called clusters. This process of tasks is called container orchestration. ECS and EKS are designed to help these orchestrations. ECS helps to run containerized applications without need of our own orchestration software. EKS also does similar with different tools. 
#### Fargate
But again these two services run over EC2 instance. We need to go serverless, then AWS offers **Fargate**. It is a serverless compute platform for ECS and EKS. In general, say we need to host applications and want access to that underlying os like windows, linux then we have to go for EC2 instance. But we want to host some short running functions or even-driven and we dont need to knw about underlying environment, we go for AWS lambda. Similarly we choose ECS and EKS and then choose to go with EC2 or Fargate.
##### Docker vs Kubernetes
> Difference between docker and kubernetes is, docker is foundation of containerization and provides a way to package and deploy applications, while Kubernetes builds on top of Docker and provides tools for managing and orchestrating multiple containers across multiple hosts.
## Infrastructure
AWS data centers are built in large groups called **Regions** like Ohio, Hyderabad etc. In each region, there are several **Availability Zones** or AZs and each AZ with one or several data centers. 
### Regions
Each region is connected with every other region through a fiber network around the globe. We can pick any region we need. Choosing a region depends on 4 major factos.
1. Compliance 
If there are any strict restrictions like data should reside in a particular region, then there is no luxury of choosing other regions, we have to abide by restrictions.
2. Proximity 
If there aren't any restrictions, then next big factor is how speed. Nearest regions to our customer base gives best speeds. Thus choosing the region in proximity of our largest customer base is ideal.
3. Feature Availability 
Few regions might not have a few new features that we desperately need. Then we have to look for another nearest region with those features.
4. Pricing 
Pricing depends on the region. If none of above factors doesn't matter for you, then go for a region that has lower prices. But there will be trade-offs w.r.t speed, availability etc..
### AZ
When we run an EC2 instance, a virtual machine is launched on a physical hardware in some AZ. If there was any natural disaster or other uncontrollable factor, all data available in an AZ might be lost. To prevent such cases, AZs in a region are built far from each other and running more than one instance will result in machines in different locations saving our data.
### Edge Locations
Say our customers are in Ohio but our data is hosted in Paris region. Instead of all requests going to Paris, we can have a copy of the data locally in Ohio. Caching copies of data closer to customers all over the world is done using **Content Delivery Networks**(CDNs). AWS CDN service is called **Amazon CloudFront**. CloudFront service uses **Edge Locations** to accelerate communication with users. These edge locations are seperate from regions, so you can push content from inside a Region to a collection of Edge locations around the world, in order to accelerate communication and content delivery. AWS Edge locations, also run more than just CloudFront. They run a domain name service, or DNS, known as **Amazon Route 53**, helping direct customers to the correct web locations with reliably low latency.

AWS can also install a fully operational mini Region, right inside our own data center called AWS **Outposts**. That's owned and operated by AWS, using 100% of AWS functionality, but isolated within our own building. It's not a solution most customers need, but if we have specific problems that can only be solved by staying in our own building, AWS Outposts can help. 

## Networking
As there are millions of customers who use AWS services and vast number of resources customers created, there should be boundaries around resources such a way that network traffic would be able to flow between them unrestricted. AWS offers **Virtual Private Cloud**(VPC) to establish boundaries around AWS resources. Amazon VPC enables us an isolated section of AWS cloud. In that section, we organize our resources into subnets. To allow public traffic from internet to access our VPC, we attach an **internet gateway** to the VPC. Similarly we have only private resources in VPC, to grant access to required users, we use a **Virtual Private Network**(VPN). VPN uses same path as VPC except our traffic is encrypted. As we use the same connection as VPC, there might be slow downs. So AWS offers one more service **Direct Connect**. It provides a dedicated connection to our VPC.  This helps us to reduce network costs and increase the bandwidth.
![AWS Direct Cloud](direct_connect.png)
What happens when a customer clicks on our application say a website? 

Once a customer reuests data drom application, Amazon **Route 53** uses DNS resolution which gets IP address through the domain name and then the customer request is sent to nearest edge location (a data centre string cache data for low latency) through Amazon **CloudFront**. The Amazn CloudFront connects to the appliction load balancer which sends the incoming packets to EC2 instance.
## Storage and Databases
While we are using EC2 instance, the virtuaal server comes with compute (CPU), memory, network and storage. Looking into the storage part, it is a block-level storage which means the storage is divided into blocks of data and overwrtitng a file will not overwrite all other blocks. Generally the block-level storages are stored in harddrive and EC2 instance comes with a harddrive too. We we run an EC2 instance, it comes with a local storage called **instance store volumes**. Well, EC2 insatnces are virtual and we will not be running on same host always. Everytime we start an EC2 insatnce, it will be running on a different host which means everytime our hardrive data will be deleted. So its only a temporary cached storage. 
### Elastic Block Store (EBS)
Amazon EBS is used to store the data that is needed outside an EC2 instance lifecycle. With EBS, we can create virtual hard drives that are called **EBS volumes**. These are seperate drives outside of our EC2 insatnce and are connected to our instance when needed. We can define the size, type and configurations of the volume we need. To be more safe, we can take incremental backups (only new and modified data) of our data called *snapshots*. 
### Simple Storage Service (S3)
While EBS provides low latency and consistent performace for EC2 instances, we need something to store different types of data to be more precise *unstructured data* and something that is highly durable, scalable and accessable.
> S3 is an object storage service that is designed for storing and retrieving unlimited amounts of unstructured data, such as images, videos, log files, and backups.

In object storage, each object contains data, metadata and a key. Metadata containes information about the data. S3 stores them in *buckets* which is like a directory for objects. The maximum object size is 5TB. We can even have version objects. S3 has different tires. 
1. S3 Standard 

An object stored in S3 standard is highly durable which means it remains intact even after a period of one year. Also data is stored in two seperate storage facilities. Another use case is static website hosting. We can upload all HTML files, static web assets and then host as static website. 
2. S3 Infrequent Access(IA) 

This is used for the data which is accessed less frequently but requires rapid access when needed something like backups, recovery files that require long term storage.
3. S3 Glacier 

It is used to archive the data. We use it for the data that is to be stored for long time but doesn't need rapid access. We can even set vault lock policy and set controls like write once/read many (WORM) which restricts from editing. We can even have lifecycle policies which will move the data automatically between tires based on the duration. Further tires available in Glacier based on the amount of time to retrieve the objects.
4. S3 Intelligent-Tiering

Ideal for data with unknown and changing access patterns. AWS will automatically move the data between tires based on access patterns.
#### EBS vs S3
S3 can be used for applications where we might need web access, URL based, high durability and also serverless without the need of EC2 instance. Ex: Most of the machine learning applications like disease detection web app or similar object finder etc. So, S3 is used where we use **complete objects and occasional changes** are needed.
While EBS is used for changing huge files as EBS uses block-level storage unlike S3 all in one object style. Ex: Video editing. So EBS is used where we need **complex read, write, change functions**.
### Elastic File System (EFS)
In **file storage**, multiple clients (such as users, applications, servers, and so on) can access data that is stored in shared file folders. In this approach, a storage server uses block storage with a local file system to organize files. Clients access data through file paths. Compared to block storage and object storage, file storage is ideal for use cases in which a large number of services and resources need to access the same data at the same time. Ex: Multiple servers running analytics on large data stored in shared file system. 
> EFS is a scalable file system used with AWS Cloud services and on-premises resources. It grows and shrinks automatically.
#### EFS vs EBS
Just like EFS has data that can be accessed by EC2 insatnces, we saw even EBS does the same. The difference is, EBS is an *Availabilty Zone-level* resource so to attach an EBS to EC2, we need to be in same availability zone (physical data center). And also, EBS doesn't scale automatically. Its nothing but a **virtual hard disk**. Whereas EFS is a **true file system** for Linux, it can have multiple insatnces reading and writing from it at the same time. As we write more data, it scales automatically. Also it is regional based instance (multiple availabilty zones in a single region).
### Relational Database Service (RDS)
In a relational database, data is stored in a way that relates it to other pieces of data. In layman terms, realtional database is a bunch of tables with relations between them. Check [here](https://github.com/nvmcr/Blog/tree/main/DBMS) to know more about relational and nonrelational databases. AWS supports many database systems like MySQL, PostgreSQL, Oracle, Mircosoft SQL Server and MariaDB. We can migrate our database from on-premise to cloud easily using **Lift-and-Shift**. The other option is RDS.
> RDS is a service that enables us to run relational databases on AWS cloud.

It supports before mentioned database systems plus more backup and security options. AWS also comes with its own database system called **Amazon Aurora** which is compatible with MySQL and PostgreSQL but 5 and 3 times faster than those two database systems relatively. Aurora replicates six copies of your data across three Availability Zones and continuously backs up your data to Amazon S3.
### DynamoDB
DynamoDB is a non-relational and serverless database. We create tables where we store and query data but instead of rows and cols, data is organized using structures like key-value pairs similar to dictionaries. Instead of having rigid schemas like relational databases, DynamoDB has a flexible schema and queries are simple and doesn't span multiple tables. So they have quick response time and highlt scalable.
> DynamoDB is a key-value serverless database service with high scalability and performance.
### Redshift
> Redshift is a massively scalable data warehousing service that can be used for big data analytics.
#### Database vs Data Warehouse vs Data Lake
A database, data warehouse, and data lake are all types of data management systems, but they differ in their design, purpose, and usage.

A database is a system that stores and manages structured data in a **highly organized manner**. Databases are typically used for transactional processing, which involves frequent read and write operations on small amounts of data. Databases use a schema to define the structure of the data and enforce consistency and integrity of the data.

A data warehouse, on the other hand, is a system that stores and manages **large volumes of historical data from multiple sources**. Data warehouses are typically used for analytical processing, which involves complex queries and analysis of large amounts of data. Data warehouses use a schema to define the structure of the data and transform the data into a format that is optimized for analytics and reporting.

A data lake is a system that stores and manages large volumes of **raw, unstructured, and semi-structured data from multiple sources**. Data lakes are designed to store data in its original form and enable a variety of data analysis and processing tasks, such as data exploration, machine learning, and advanced analytics. Data lakes do not enforce a rigid schema, and the data can be transformed and processed in a variety of ways.

In summary, a database is a system that manages structured data for transactional processing, a data warehouse is a system that manages large volumes of historical data for analytical processing, and a data lake is a system that manages large volumes of raw data for exploratory and analytical processing. Each type of data management system has its own strengths and weaknesses, and the choice of system depends on the specific use case and requirements.
### Additional Database Services
We might need different database for different purposes. So AWS offers a wide range of databases.
#### DocumentDB
DocumentDB is a document database service that supports MongoDB workloads. (MongoDB is a document database program.)
#### Neptune
Neptune is a graph database service. We can use Amazon Neptune to build and run applications that work with highly connected datasets, such as recommendation engines, fraud detection, and knowledge graphs.
#### Quantum Ledger Database 
QLDB is a ledger database service. We can use Amazon QLDB to review a complete history of all the changes that have been made to your application data.
#### Managed Blockchain 
It is a service that you can use to create and manage blockchain networks with open-source frameworks. Blockchain is a distributed ledger system that lets multiple parties run transactions and share data without a central authority.
#### DynamoDB Accelerator 
DAX is an in-memory cache for DynamoDB. It helps improve response times from single-digit milliseconds to microseconds.

