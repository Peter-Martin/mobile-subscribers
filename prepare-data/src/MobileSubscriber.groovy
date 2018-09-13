/**
 * Represents an UNLABELLED Mobile Subscriber,
 * includes logic for fixing invalid feature values.
 */

import groovy.transform.ToString

import java.util.regex.Pattern

//@ToString(includePackage = false, includeNames=true) // names useful for debugging
@ToString(includePackage = false)
class MobileSubscriber {

    // use max (1) or reduce by random multiplier (0)
    static float[] USAGE_MULTIPLIERS = [1, 0, 0, 0]
    static Random umRandomSelector = new Random()

    // group voice/sms/data in purchase price bands
    // so we can adjust subscribers' quantities based on how much they've actually paid
    static def PURCHASED_PRICE_BANDS = [
        [voice: 0,     sms: 0,     data: 0],
        [voice: 100,   sms: 250,   data: 1000],
        [voice: 500,   sms: 250,   data: 5000],
        [voice: 500,   sms: 10000, data: 10000],
        [voice: 10000, sms: 10000, data: 100000],
    ]

    // we generate CSV by just stripping out unwanted text from default toString()
    static Pattern CSV_PATTERN = Pattern.compile('.*?(-*\\d+,*)(\\)*).*?')
    static String CSV_REPLACEMENT = '$1'

    // NB: field order is important here; it's used when outputting CSV

    String Postpaid
    String ContractLengthWeeks

    String AveragePayment
    String NumberOfPayments
    String AveragePeakCreditBalance

    String VoicePurchasedMinutesPerMonth
    String VoiceUsedMinutesPerMonth
    String VoiceNumberOfCallsPerMonth

    String SmsPurchasedPerMonth
    String SmsUsedPerMonth

    String DataMbPurchasedPerMonth
    String DataMbUsedPerMonth

    // can't use more voice/sms/data than that purchased, force used <= purchased
    private static String fixUsageValues(String usedStr, String purchasedStr) {
        int used = usedStr as int
        int purchased = purchasedStr as int
        if (used > purchased) {
            float usageMultiplier = USAGE_MULTIPLIERS[umRandomSelector.nextInt(USAGE_MULTIPLIERS.size())]
            if (usageMultiplier == 0) usageMultiplier = umRandomSelector.nextFloat() // rand between 0.0 and 1.0
            int fixedUsage = (purchased * usageMultiplier) as int
            "$fixedUsage"
        } else {
            "$used"
        }
    }

    // adjust purchased quantities if not reflected by what subscriber actually paid
    private void fixPurchasedValues() {
        int purchasedVoice = VoicePurchasedMinutesPerMonth as int
        int purchasedSms = SmsPurchasedPerMonth as int
        int purchasedData = DataMbPurchasedPerMonth as int
        int averagePayment = AveragePayment as int
        int numberOfPayments = NumberOfPayments as int
        // index (into PURCHASED_PRICE_BANDS) that shows what subscriber *should* receive
        int purchasedPriceBandIndex = ((averagePayment * numberOfPayments) / 120) as int
        // if paid > 0, then can't be in zero price band, so bump them up to minimum
        if (averagePayment > 0 && purchasedPriceBandIndex == 0) purchasedPriceBandIndex = 1
        if (purchasedPriceBandIndex >= PURCHASED_PRICE_BANDS.size()) {
            purchasedPriceBandIndex = PURCHASED_PRICE_BANDS.size() - 1
        }
        def purchasedPriceBand = PURCHASED_PRICE_BANDS[purchasedPriceBandIndex]
        int compareToPurchased = (
            purchasedPriceBand.voice.compareTo(purchasedVoice) +
            purchasedPriceBand.sms.compareTo(purchasedSms) +
            purchasedPriceBand.data.compareTo(purchasedData)
        )
        // if getting more/less than they should in 2 out of 3 of voice/sms/data,
        // then adjust their purchased levels to correct price band values
        if (compareToPurchased >= 2 || compareToPurchased <= -2) {
            VoicePurchasedMinutesPerMonth = purchasedPriceBand.voice
            SmsPurchasedPerMonth = purchasedPriceBand.sms
            DataMbPurchasedPerMonth = purchasedPriceBand.data
        } // else just leave values as they are
    }

    // force values to sensible ones
    MobileSubscriber fix() {

        fixPurchasedValues()

        // fix usage values, can't be > purchased: common across prepaid/postpaid
        VoiceUsedMinutesPerMonth = fixUsageValues(VoiceUsedMinutesPerMonth, VoicePurchasedMinutesPerMonth)
        SmsUsedPerMonth = fixUsageValues(SmsUsedPerMonth, SmsPurchasedPerMonth)
        DataMbUsedPerMonth = fixUsageValues(DataMbUsedPerMonth, DataMbPurchasedPerMonth)
        // have to tally voice usage versus number of calls
        if (VoiceUsedMinutesPerMonth == '0') {
            // zero voice usage => must be zero calls
            VoiceNumberOfCallsPerMonth = 0
        } else if (VoiceNumberOfCallsPerMonth == '0') {
            // some voice usage => can't be zero calls, so just average to 15 mins per call
            VoiceNumberOfCallsPerMonth = ((VoiceUsedMinutesPerMonth as int) / 15) as int
        }
        // further specific adjustments for prepaid/postpaid
        switch(Postpaid) {
            case '0': // prepaid
                ContractLengthWeeks = '0' // no contract
                // purchased calls/sms/data same as used; i.e., no advance purchase
                VoicePurchasedMinutesPerMonth = VoiceUsedMinutesPerMonth
                SmsPurchasedPerMonth = SmsUsedPerMonth
                DataMbPurchasedPerMonth = DataMbUsedPerMonth
                break

            case '1': // postpaid
                // can't be zero-length contract, min 1 month
                if (ContractLengthWeeks == '0') ContractLengthWeeks = '4'
                // average payment should be 10/20/40; i.e., fixed values
                int averagePayment = AveragePayment as int
                AveragePayment = 10 * (2 ** (averagePayment / 20) as int)
                // number of payments always 12 (pay by month)
                NumberOfPayments = '12'
                // no 'credit' for postpaid, so APCB always = AP
                AveragePeakCreditBalance = AveragePayment
                break
        }
        this
    }

    // generate CSV by just stripping out unwanted text from default toString()
    String toCsvString() {
        this.toString().replaceAll(CSV_PATTERN, CSV_REPLACEMENT)
    }
}
