# Source-member recovery during the blinded header inventory

At the October 20, 09:45 UTC checkpoint, four extracted DICOM objects in two examinations differed from the source inventory. Repeated reads reproduced the differences. Both saved ZIP archives matched their acquisition SHA-256 receipts and recorded object-metadata MD5 values. The corresponding archive members passed their ZIP CRC checks and matched the frozen inventory SHA-256 values. The object-metadata checksums are uploader-recorded values, not independently signed source attestations.

The four original extracted objects were preserved. Copies of the matching archive members were written to a new output area, and a new input overlay points to those copies while retaining the remaining input paths. The four recovered members total 501,546 bytes. This restores the intended inventory bytes; it does not change a clinical eligibility rule, target, model or endpoint. The cause of the changes to the extracted objects has not been established.

The next version of the header job records the overlay and prior-plan identities. It reuses 21,522 source-verified header rows with copy lineage and rereads the four failed objects. The first two resumed shards contain 21,526 records with zero identity failures, including 120 authenticated parse failures. Remaining shards are still being collected. Parse failures include source objects that are not valid DICOM; they do not automatically exclude a clinical examination.

Header shards run independently with one drive reader, so an error in one shard does not prevent the other shards being collected. The complete cache and technical packet still require every source disposition to reconcile. No reference outcome rows or model predictions were used in this recovery.

Internal disk pressure also required moving this run's header receipts to Research. All 72 files, totaling 112,756,811 logical bytes, were compared by size and SHA-256 before the local copies were removed. Stable local links preserve their references. Generated exFAT sidecars were recorded separately. Source images, archives and historical batches were not moved or overwritten by that operation.

Private evidence includes the four-member diagnostic lock/result, overlay receipt, authenticated-row reuse receipt, migration receipt and separate versioned execution logs. These document technical input recovery; they do not establish external clinical performance.
