Attribute VB_Name = "Módulo2"
Sub Create_Pivot_Table_Compra_normal()
'Todas las lineas de solicitudes que:
'No sean catalogos
'No COE

'https://www.thespreadsheetguru.com/blog/2014/9/27/vba-guide-excel-pivot-tables
'http://www.globaliconnect.com/excel/index.php?option=com_content&view=article&id=150:excel-pivot-tables-filter-data-items-values-a-dates-using-vba&catid=79&Itemid=475

Mensaje = MsgBox("Debe tener actualizada la planilla base" & vbCrLf & "Tipo de compra: Sourcing" & vbCrLf & "Pais: Chile y Peru" & vbCrLf & "No COE" & vbCrLf & "Cantidad de lineas: se filtra 1 que indica la cantidad de OC" & vbCrLf & "Esta tabla dinamica muestra las solicitudes pendientes de PO que no son catalogos, contratos ni COE.", vbOKCancel)
If Mensaje = vbCancel Then
Exit Sub
End If


Call tipo_de_compra_BASE

'Declare Variables
Dim PSheet As Worksheet 'To create a sheet for a new pivot table.
Dim DSheet As Worksheet 'To use as a data sheet.
Dim PCache As PivotCache 'To use as a name for pivot table cache.
Dim PTable As PivotTable 'To use as a name for our pivot table.
Dim PRange As Range 'to define source data range.
Dim lastRow As Long 'To get the last row and column of our data range.
Dim LastCol As Long 'To get the last row and column of our data range.

'Insert a New Blank Worksheet
On Error Resume Next
Application.DisplayAlerts = False
Worksheets("Compra normal").Delete
Sheets.Add Before:=ActiveSheet
ActiveSheet.Name = "Compra normal"
Application.DisplayAlerts = True
Set PSheet = Worksheets("Compra normal")
Set DSheet = Worksheets("Base")

'Define Data Range
lastRow = DSheet.Cells(Rows.Count, 1).End(xlUp).Row
LastCol = DSheet.Cells(5, Columns.Count).End(xlToLeft).Column
Set PRange = DSheet.Cells(5, 1).Resize(lastRow, LastCol)

'Define Pivot Cache
Set PCache = ActiveWorkbook.PivotCaches.Create _
(SourceType:=xlDatabase, SourceData:=PRange). _
CreatePivotTable(TableDestination:=PSheet.Cells(2, 2), _
TableName:="Compra_normal")

'Insert Blank Pivot Table
Set PTable = PCache.CreatePivotTable _
(TableDestination:=PSheet.Cells(1, 1), TableName:="Compra_normal")

'Insert Report Filter (estos son los filtros)

With ActiveSheet.PivotTables("Compra_normal").PivotFields("Compra realizada")
.Orientation = xlPageField
.PivotItems("OC No Realizada").Visible = True
.PivotItems("OC Realizada").Visible = False
.PivotItems("(blank)").Visible = False
.Position = 5
End With


With ActiveSheet.PivotTables("Compra_normal").PivotFields("Cantidad de lineas")
.Orientation = xlPageField
.PivotItems("1").Visible = True
        For i = 2 To .PivotItems.Count
            With .PivotItems(i)
                If i > 1 Then
                    .Visible = False
                End If
            End With
        Next i
.Position = 4
End With

With ActiveSheet.PivotTables("Compra_normal").PivotFields("Pais")
.Orientation = xlPageField
.PivotItems("Chile").Visible = True
.PivotItems("Perú").Visible = True
.Position = 3
End With

With ActiveSheet.PivotTables("Compra_normal").PivotFields("Tipo de compra")
.Orientation = xlPageField
.PivotItems("Catalogo").Visible = False
.PivotItems("Contrato").Visible = False
.PivotItems("Sourcing").Visible = True
.PivotItems("Politica").Visible = True
.PivotItems("(blank)").Visible = False
.Position = 2
End With

With ActiveSheet.PivotTables("Compra_normal").PivotFields("Area de compra")
.Orientation = xlPageField
.PivotItems("COE").Visible = False
.PivotItems("Compra").Visible = True
.PivotItems("Gestion PO").Visible = True
.PivotItems("(blank)").Visible = False
.Position = 1
End With


'Insert Column Fields  (estos son las columnas)
With ActiveSheet.PivotTables("Compra_normal").PivotFields("Dias Pen")
.Orientation = xlColumnField
.Position = 1
.PivotItems("").Visible = False
.PivotItems("(blank)").Visible = False
.Subtotals(1) = False
End With

'Insert Row Labels (estos son las filas)
With ActiveSheet.PivotTables("Compra_normal").PivotFields("Taxonomia")
.Orientation = xlRowField
.PivotItems("").Visible = False
.PivotItems("(blank)").Visible = False
.PivotItems("OC Realizada").Visible = False
.Position = 2
.Subtotals(1) = False
End With

With ActiveSheet.PivotTables("Compra_normal").PivotFields("Clasificacion categoria")
.Orientation = xlRowField
.PivotItems("Infraestructura").Visible = True
.PivotItems("Servicios").Visible = True
.PivotItems("Mkt").Visible = True
.PivotItems("Salud - Office").Visible = True
.PivotItems("Gestión PO").Visible = True
.PivotItems("").Visible = False
.PivotItems("(blank)").Visible = False
.Position = 1
.Subtotals(1) = True
End With



'Insert Data Field (estos son las valores)
With ActiveSheet.PivotTables("Compra_normal").PivotFields("Lineadistribucion")
.Orientation = xlDataField
.Position = 1
.Function = xlCount
.NumberFormat = "#,##0"
.Subtotals(1) = False
End With




'http://www.globaliconnect.com/excel/index.php?option=com_content&view=article&id=150:excel-pivot-tables-filter-data-items-values-a-dates-using-vba&catid=79&Itemid=475

'Format Pivot Table
'ActiveSheet.PivotTables("SalesPivotTable").ShowTableStyleRowStripes = True
'ActiveSheet.PivotTables("SalesPivotTable").TableStyle2 = "PivotStyleMedium9"

End Sub

