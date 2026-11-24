# Direct Debit Failures

Acme Payments Knowledge Base · Payments Operations

## Overview

A direct debit allows a merchant to collect money directly from a customer's bank account using a mandate the customer has approved in advance. Most direct debits complete without problems, but a small percentage fail every day. When a debit fails, no money moves, the merchant's collection is marked as failed, and the customer may be notified depending on the merchant's settings.

This article explains why direct debits fail, what each failure code means, how automatic retries work, and what merchants and support agents should do next.

## Common reasons a direct debit fails

### Insufficient funds

The most common cause of failure is that the customer's account does not have enough money to cover the debit at the moment it is presented to the bank. Banks check the available balance, not the ledger balance, so pending card transactions or holds can reduce the available amount even when the customer believes the money is there.

### Inactive or cancelled mandate

A debit can only be collected against an active mandate. If the customer has cancelled the mandate, if the mandate has expired, or if it was never fully approved by the customer's bank, the debit is rejected before it reaches the customer's account. Mandates that have not been used for 13 consecutive months are automatically marked as dormant and must be re-authorised before new debits can be collected.

### Account closed or blocked

If the customer's account has been closed, frozen by the bank, or placed under a regulatory block, the debit is rejected. These failures cannot be fixed by retrying; the customer must provide new account details and approve a new mandate.

### Bank rejection

The issuing bank may reject a debit for internal reasons, such as suspected fraud, a temporary system outage, or a mismatch between the mandate details and the account holder's records. Bank rejections are sometimes temporary, so they are eligible for automatic retries.

### Amount exceeds the mandate limit

Every mandate has a maximum amount per debit and, optionally, a maximum number of debits per month. A debit that exceeds either limit is rejected immediately. The merchant must either reduce the amount, split the collection into smaller debits within the monthly limit, or ask the customer to approve a new mandate with a higher limit.

## Failure codes

| Code | Meaning | Retry allowed? |
|------|---------|----------------|
| DD01 | Insufficient funds | Yes |
| DD02 | Account closed or blocked | No |
| DD03 | Mandate inactive, cancelled, or dormant | No |
| DD04 | Bank rejection (general) | Yes |
| DD05 | Amount exceeds mandate limit | No |
| DD06 | Invalid account details | No |
| DD07 | Bank system unavailable | Yes |

## Automatic retries

Failures with a retryable code (DD01, DD04 and DD07) are retried automatically. Acme Payments retries a failed debit up to two times, with each retry scheduled three business days after the previous attempt. Retries are always presented in the morning processing window, because customers are more likely to have received salary payments by then.

If the final retry also fails, the collection is marked as permanently failed and no further attempts are made. Merchants can still trigger a manual retry from the dashboard, but manual retries count towards the monthly debit limit on the mandate.

Retries are never attempted for non-retryable codes. Retrying a debit against a closed account or a cancelled mandate will always fail and may lead to the merchant being flagged by the customer's bank.

## Customer notifications

When a debit fails, the customer receives an SMS and an email within 15 minutes, provided the merchant has enabled failure notifications. The message includes the merchant name, the amount, the reason for the failure in plain language, and the date of the next automatic retry if one is scheduled.

Merchants can customise the wording of these messages but cannot remove the failure reason, because customers have a right to know why a payment was not collected.

## What merchants should do

1. Check the failure code in the dashboard or in the `collection.failed` webhook.
2. For retryable codes, wait for the automatic retries before contacting the customer.
3. For non-retryable codes, contact the customer and ask them to update their account details or approve a new mandate.
4. Do not create a new debit for the same invoice while automatic retries are still scheduled, as this can result in the customer being charged twice.

## Escalation

If a merchant reports that debits are failing for many customers at the same bank at the same time, treat it as a possible bank outage rather than individual customer problems. Support agents should check the status page and escalate to the Payments Operations on-call engineer if the failure rate for a single bank exceeds 20% in any one-hour period.
