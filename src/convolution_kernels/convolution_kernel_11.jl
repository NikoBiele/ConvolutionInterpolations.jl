# see 'docstring.jl' for documentation
function (::ConvolutionKernel{11})(s::T) where {T} # 8 equation 7th order accurate 11th degree
    s_abs = abs(s)
    coef = Dict(
        # 8 equation 11th degree, 7th order accurate
        :eq1 => [1, 0, -1474545903118565939/792228434572753920, 0, 78507043213210069/66019036214396160, 0, -12670369340107457/31437636292569600, 0, 46165786466951987/528152289715169280, 0, -1979989382272480301/126756549531640627200, 12125765928110303/3621615700904017920], 
        :eq2 => [46688025415070421283/47533706074365235200, 611754868632362851/3521015264767795200, -12454722519126817579/4753370607436523520, 1763170764321368369/905403925226004480, -360348012477965893/176050763238389760, 8167009247425360189/2263509813065011200, -873960348562623487/282938726633126400, 266121542973409453/211260915886067712, -675120327792804377/3168913738291015680, -183362958223076599/19013482429746094080, 1047951963907767427/126756549531640627200, -311721508838343637/380269648594921881600],
        :eq3 => [290789554778345414497/15844568691455078400, -1176286025464014790457/13581058878390067200, 914864288891214509429/4753370607436523520, -326860792073164279381/1267565495316406272, 356420440040563131203/1584456869145507840, -299498596129286544419/2263509813065011200, 279727167851339131/5239606048761600, -46925594811500374541/3168913738291015680, 880436123772301723/316891373829101568, -6412971600229340729/19013482429746094080, 9103327328716715587/380269648594921881600, -287163581109902653/380269648594921881600],
        :eq4 => [-97624152288495515653/282938726633126400, 28487486459475436646677/23766853037182617600, -1270466651679226806229/679052943919503360, 2736179354334916916131/1584456869145507840, -59178644728015058323/56587745326625280, 246848921284187478049/565877453266252800, -2682038553642224969/20958424195046400, 4187717595896072111/158445686914550784, -1707133950733157057/452701962613002240, 3369716421265306457/9506741214873047040, -1074497928967420859/54324235513560268800, 188990552404159519/380269648594921881600],
        :eq5 => [-91686872256601949/573247781890560, 584032665663098652419/3395264719597516800, 93995742861929057039/1584456869145507840, -65298785669132272249/316891373829101568, 7187902958433605347/44012690809597440, -4519515363364915111/62875272585139200, 3826511287881600271/188625817755417600, -607682890871518961/158445686914550784, 513414309069248863/1056304579430338560, -378248565674872117/9506741214873047040, 80587128202093921/42252183177213542400, -3100113788719121/76053929718984376320],
        :eq6 => [11400592501538893097803/3168913738291015680, -678700264659088852966643/95067412148730470400, 30492761304819166077367/4753370607436523520, -624031058955709345303/181080785045200896, 1948924850418375808117/1584456869145507840, -693426374210262490871/2263509813065011200, 854130808285056533/15718818146284800, -4351077069530586469/633782747658203136, 468519144160859/773660580637455, -675477219960947609/19013482429746094080, 474432619668728489/380269648594921881600, -1511433282576589/76053929718984376320],
        :eq7 => [1629996666173636922641/2263509813065011200, -111462464545403697599219/95067412148730470400, 4119729807398281352407/4753370607436523520, -2433650625602123990309/6337827476582031360, 179520527300474443621/1584456869145507840, -52918948738542222743/2263509813065011200, 108230575125344359/31437636292569600, -32757522364242595/90540392522600448, 84273445110240251/3168913738291015680, -24758575515423287/19013482429746094080, 14536864713060107/380269648594921881600, -193847014686169/380269648594921881600],
        :eq8 => [6324204691423232/1657844101365975, -20553665247125504/3868302903187275, 7806440165975552/2320981741912365, -197631396606976/154732116127491, 3087990571984/9551365193055, -31651903362836/552614700455325, 16018951092167/2210458801821300, -192999410749/294727840242840, 3280989982733/79222843457275392, -8298974662207/4753370607436523520, 5596982911721/126756549531640627200, -192999410749/380269648594921881600],
    )
    if s_abs < 1.0
        return horner(s_abs, coef, :eq1)
    elseif s_abs < 2.0
        return horner(s_abs, coef, :eq2)
    elseif s_abs < 3.0
        return horner(s_abs, coef, :eq3)
    elseif s_abs < 4.0
        return horner(s_abs, coef, :eq4)
    elseif s_abs < 5.0
        return horner(s_abs, coef, :eq5)
    elseif s_abs < 6.0
        return horner(s_abs, coef, :eq6)
    elseif s_abs < 7.0
        return horner(s_abs, coef, :eq7)
    elseif s_abs < 8.0
        return horner(s_abs, coef, :eq8)
    else
        return 0.0
    end
end