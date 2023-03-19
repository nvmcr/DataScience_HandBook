# Cloud Essentials
## Table of Contents
1. [Intro](#Intro)
    1. [Client-Server Model](#client-server-model)
    2. [Cloud Computing](#cloud-computing)
2. [Elastic Compute Cloud (EC2)](#EC2)
    1. [EC2 Instance Types](#types-of-ec2-instances)
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
