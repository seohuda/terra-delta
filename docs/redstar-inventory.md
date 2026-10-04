# Redstar read-only inventory, 2026-10-04

Scope: `i-0426ce1098082fee2`, `ap-northeast-2`, resources linked by attachment,
dedicated VPC or explicit Redstar project tags. No deletion, start, resize or
configuration change was made. Existing services and their addresses are KEEP.

| Resource | Observed state | Classification / dependency |
| --- | --- | --- |
| EC2 `i-0426ce1098082fee2` | `c8i.xlarge`, stopped; `Project=redstar-server` | REVIEW before terminate: root EBS would be deleted. User confirms project unused. |
| EBS `vol-0fd71ecf4d0529fc7` | 80 GB gp3, 3,000 IOPS / 125 MB/s, unencrypted; attached, DeleteOnTermination=true | REVIEW: contains uninspected project data; preserve until data-retention decision. |
| Owned snapshots | None found by source volume or Redstar Project tag | No snapshot to remove. The source AMI snapshot ID on the volume is not evidence of an owned backup. |
| EIP | None associated with Redstar EC2/ENI or explicitly Redstar tagged | No release candidate. Other-service EIPs KEEP. |
| ENI `eni-099e0802852b4863c` | Primary ENI, attached, DeleteOnTermination=true | SAFE_TO_DELETE with EC2 after root data is preserved; do not manually detach a primary interface. |
| SG `sg-0bd4183a54395e59a` | Redstar SG; used only by the observed Redstar ENI; no inbound SG references found | SAFE_TO_DELETE candidate after EC2/ENI removal and renewed dependency check. |
| IAM instance profile | No profile present on instance description | None to remove. |
| ELB / target groups | No regional classic/application/network LB or target group returned | No observed association. |
| Route53 | No record containing Redstar name or current private IP in the two returned zones | KEEP zones and unrelated records; historical released public-IP references cannot be inferred from this scan. |
| NAT / VPC endpoints | None in Redstar VPC | No observed recurring NAT/endpoint cost. |
| VPC `vpc-02bfac1d2c2c99aee` | Dedicated Redstar tags; one Redstar ENI / subnet | SAFE_TO_DELETE candidate after EC2 and dependencies removed. Shared DHCP options KEEP. |
| Subnet `subnet-0464edc6f3440a0b6` | Dedicated Redstar public subnet | SAFE_TO_DELETE candidate after ENI removal. |
| IGW `igw-0aeee0999279273d5` | Attached only to Redstar VPC, tagged | SAFE_TO_DELETE candidate after public-address/ENI dependencies gone. |
| Route table `rtb-072cc6f6174ea4fde` | Redstar public table, associated with Redstar subnet | SAFE_TO_DELETE candidate after disassociation. Default main route table is managed through VPC deletion. |
| Key pair `redstar-server-key` | Redstar project/name tags | SAFE_TO_DELETE candidate after retaining any needed recovery access. No storage cost saving. |

Tag inventory returned eight project resources: instance, EBS, SG, VPC, subnet,
IGW, route table and key pair. No separately tagged orphan EBS was returned.
This is a scoped inventory, not proof that every untagged/global dependency is
absent. Recheck before executing any eventual cleanup.

Live AWS Price List API: Seoul gp3 baseline storage **USD 0.0912/GB-month**,
so 80 GB is **USD 7.296/month** before taxes, discounts or account credits.
Stopped compute is not charged for active instance hours; the attached EBS
continues to exist. No extra provisioned IOPS/throughput above gp3 baseline was
observed. Cost Explorer tag query for 2026-09-04 through 2026-10-03 returned empty
groups, but `Project` is **Inactive** as a cost-allocation tag. Therefore that
query does not establish zero actual Redstar cost; actual attribution is unknown.

Cleanup plan, not executed:

1. Decide retention of the uninspected 80 GB root data. Before any terminate,
   either preserve that EBS by setting DeleteOnTermination=false or confirm a
   verified backup and its restoration requirements.
2. Terminate the unused EC2 only after the data decision; verify ENI removal.
3. Review any retained/orphan EBS and owned snapshots separately; no data-volume
   deletion is implied by the user's statement that the project is unused.
4. Recheck and remove dedicated unused SG, subnet, route-table association,
   IGW, VPC and key pair in dependency order if authorized.
5. Do not release other-service EIPs or modify bot-server, TripPoint, Satulingo,
   VTuber, SAM or other project resources.
