/**
 * Represents a LABELLED Mobile Subscriber,
 * includes rules for applying labels according to feature values.
 */

import groovy.transform.ToString

//@ToString(includePackage = false, includeNames=true) // names useful for debugging
@ToString(includePackage = false)
class LabelledMobileSubscriber {

    // evaluate given % random chance
    // note that won't work for percent > 50 (because we apply int rounding)
    static Random chanceRandomSelector = new Random()
    private static boolean chance(int percent) {
        if (percent == 0) return false
        if (percent == 100) return true
        int oneIn = (100 / percent) as int // convert percent to '1 in 20', etc.
        int rand = 1 + chanceRandomSelector.nextInt(oneIn)
        rand == oneIn
    }

    private MobileSubscriber sub
    // calculate/convert following values from subscriber
    private int contractLengthWeeks = 0
    private boolean isBusiness = false
    private int totalSpend = 0
    private float voiceUsedPercent = 0
    private float smsUsedPercent = 0
    private float dataUsedPercent = 0
    private int purchasedVoice = 0
    private int purchasedSms = 0
    private int purchasedData = 0
    private int averagePayment = 0
    private int numberOfPayments = 0

    // NB: field order is important here; it's used when outputting CSV

    Updated UpdatedIn90Days = Updated.UNCHANGED // unchanged plan
    int PurchasedAdditionalIn90Days = 0  // didn't purchase additional products

    boolean cancelled() {
        // if spending a lot, more likely to 'cancel' (or stop topping up for prepaid),
        // also more likely to cancel if they're business (more cost-conscious)
        int cancelPercent = 5 // 5% cancel for no good reason
        switch(sub.Postpaid) {
            case '0': // prepaid
                if (totalSpend >= 1000) {
                    cancelPercent = 10
                    if (isBusiness) cancelPercent *= 2
                }
                break
            case '1': // postpaid
                if (totalSpend >= 480) {
                    // shorter contracts more likely to cancel
                    if (contractLengthWeeks < 52) cancelPercent = 20 else cancelPercent = 10
                    if (isBusiness) cancelPercent *= 2
                }
                break
        }
        chance(cancelPercent)
    }

    boolean downgraded() { // postpaid only
        int downgradePercent = 0
        switch(sub.Postpaid) {
            case '1': // postpaid
                downgradePercent = 5 // 5% downgrade for no good reason
                // spending moderate amount, using < 70% of purchased data/sms/voice
                if (totalSpend > 120) {
                    if (voiceUsedPercent < 70 && smsUsedPercent < 70 && dataUsedPercent < 70) {
                        downgradePercent = 20
                        if (isBusiness) downgradePercent *= 2
                    }
                }
                break
        }
        chance(downgradePercent)
    }

    boolean switchedToPrepaid() { // postpaid only
        int switchedPercent = 0
        switch(sub.Postpaid) {
            case '1': // postpaid
                // 10% of downgrades are actually switches to prepaid
                switchedPercent = UpdatedIn90Days == Updated.DOWNGRADED ? 10 : 0
                break
        }
        chance(switchedPercent)
    }

    boolean upgraded() { // for prepaid, means switched to postpaid
        int upgradePercent = 2 // 2% upgrade for no good reason
        switch(sub.Postpaid) {
            case '0': // prepaid
                // if infrequent topups of moderate amount, then may go postpaid
                if (numberOfPayments <= 15 && averagePayment >= 25) {
                    upgradePercent = 20
                    if (isBusiness) upgradePercent *= 2
                }
                break
            case '1': // postpaid
                // not on top plan, but close to one of voice/sms/data limit, may upgrade
                if (totalSpend < 480) {
                    if (voiceUsedPercent >= 70 || smsUsedPercent >= 70 || dataUsedPercent >= 70) {
                        upgradePercent = 30
                        if (isBusiness) upgradePercent *= 2
                    }
                }
                break
        }
        chance(upgradePercent)
    }

    boolean purchasedAdditionalProducts() {
        int purchasedAdditionalPercent // 5% purchase additional for no good reason
        // otherwise, 10% of upgraders purchase additional products
        purchasedAdditionalPercent = UpdatedIn90Days == Updated.UPGRADED ? 10 : 5
        chance(purchasedAdditionalPercent)
    }

    // Apply label values as follows:
    //
    // UpdatedIn90Days
    //   1      upgraded plan       (for prepaid, means switched to postpaid)
    //   0      unchanged plan
    //   -1     downgraded plan     (postpaid only)
    //   -2     switched to prepaid (postpaid only)
    //   -3     cancelled           (prepaid stopped topups, postpaid cancelled contract)
    //
    // PurchasedAdditionalIn90Days
    //   1      purchased additional products
    //   0      didn't purchase additional products
    //
    LabelledMobileSubscriber label() {

        // subscriber values as ints to help subsequent calculations
        contractLengthWeeks = sub.ContractLengthWeeks as int
        purchasedVoice = sub.VoicePurchasedMinutesPerMonth as int
        purchasedSms = sub.SmsPurchasedPerMonth as int
        purchasedData = sub.DataMbPurchasedPerMonth as int
        averagePayment = sub.AveragePayment as int
        numberOfPayments = sub.NumberOfPayments as int

        // guess is business if many calls, high data usage
        isBusiness = (sub.VoiceNumberOfCallsPerMonth as int) >= 300 &&
                     (sub.DataMbUsedPerMonth as int) >= 10000

        // other calculations to help labelling
        totalSpend = numberOfPayments * averagePayment
        if (sub.VoicePurchasedMinutesPerMonth != '0') {
            voiceUsedPercent = (sub.VoiceUsedMinutesPerMonth as int) / (sub.VoicePurchasedMinutesPerMonth as int)
        }
        if (sub.SmsPurchasedPerMonth != '0') {
            smsUsedPercent = (sub.SmsUsedPerMonth as int) / (sub.SmsPurchasedPerMonth as int)
        }
        if (sub.DataMbPurchasedPerMonth != '0') {
            dataUsedPercent = (sub.DataMbUsedPerMonth as int) / (sub.DataMbPurchasedPerMonth as int)
        }

        // evaluate labels in specific order as some are exclusive of each other
        if (cancelled()) {
            UpdatedIn90Days = Updated.CANCELLED
        } else {
            if (downgraded()) {
                UpdatedIn90Days = Updated.DOWNGRADED
                if (switchedToPrepaid()) { // specific downgrade case
                    UpdatedIn90Days = Updated.SWITCHED_TO_PREPAID
                }
            } else {
                if (upgraded()) {
                    UpdatedIn90Days = Updated.UPGRADED
                }
            }
            if (purchasedAdditionalProducts()) {
                PurchasedAdditionalIn90Days = 1
            }
        }

        this
    }

    // generate CSV by just stripping out unwanted text from default toString()
    String toCsvString() {
        sub.toCsvString() + ',' +
        this.toString().replaceAll(MobileSubscriber.CSV_PATTERN, MobileSubscriber.CSV_REPLACEMENT)
    }
}

enum Updated {
    CANCELLED(0), SWITCHED_TO_PREPAID(1), DOWNGRADED(2), UNCHANGED(3), UPGRADED(4)

    int value

    Updated(int value) { this.value = value }

    @Override
    String toString() { return value }
}
